import contextlib
import json
import logging
import os
import pickle
import shutil
import socket
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
import zmq
from numpy.typing import NDArray

from pyrinst.utils.numderiv import grad_from_energy, hess_from_energy, hess_from_grad

from .base import (
    BatchPotentialResult,
    Level,
    OnTheFlyPotential,
    Potential,
    PotentialResult,
)

logger = logging.getLogger(__name__)
ExecutorResult = PotentialResult | BatchPotentialResult
PARALLEL_TIMEOUT_MS = 300000
SERVER_INFO_PATH = Path("server_info.json")


def read_server_info() -> dict:
    while True:
        if not SERVER_INFO_PATH.exists():
            time.sleep(1)
            continue
        try:
            with SERVER_INFO_PATH.open(encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError:
            time.sleep(0.2)


class Executor(ABC):
    """Apply a potential to one or many geometries."""

    @abstractmethod
    def evaluate(self, x: NDArray, level: Level = Level.GRAD) -> ExecutorResult:
        """Evaluate one or many geometries."""

    def compute(self, geom, level: Level = Level.GRAD) -> None:
        geom.V, geom.G, geom.H = self.evaluate(geom.x, level)

    @staticmethod
    def _stack_results(results: Sequence[PotentialResult]) -> BatchPotentialResult:
        if not results:
            return np.array([]), None, None
        energy = np.array([res[0] for res in results], dtype=float)
        grad = None if results[0][1] is None else np.array([res[1] for res in results])
        hess = None if results[0][2] is None else np.array([res[2] for res in results])
        return energy, grad, hess

    @staticmethod
    def _workdir(root: str, tracker: int, bead_id: int | None = None) -> str:
        if bead_id is None:
            return os.path.join(root, str(tracker))
        return os.path.join(root, str(tracker), str(bead_id))

    @staticmethod
    def _prepare_workdir(root: str, tracker: int, bead_id: int | None = None) -> str:
        workdir = Executor._workdir(root, tracker, bead_id)
        os.makedirs(workdir, exist_ok=False)
        return workdir


class SingleExecutor(Executor):
    """Executor for single-geometry evaluation."""

    def __init__(self, potential: Potential, working_dir: str = "."):
        self.potential = potential
        self._tracker = -1
        if isinstance(self.potential, OnTheFlyPotential):
            working_dir = os.path.abspath(working_dir)
            self.potential._working_dir = working_dir
            self.potential._folder = working_dir
            if working_dir != os.getcwd():
                shutil.rmtree(working_dir, ignore_errors=True)
            os.makedirs(working_dir, exist_ok=True)

    def evaluate(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult:
        if isinstance(self.potential, OnTheFlyPotential):
            self._tracker += 1
            return self._evaluate_onthefly(x, level, self._tracker)
        return self.potential(x, level)

    def _evaluate_onthefly(
        self, x: NDArray, level: Level = Level.GRAD, tracker: int | None = None, bead_id: int | None = None
    ) -> PotentialResult:
        tracker = self._tracker if tracker is None else tracker
        self.potential._folder = self._prepare_workdir(self.potential._working_dir, tracker, bead_id)
        return self.potential(x, level)


class SerialExecutor(SingleExecutor):
    """Serial executor for batch evaluation."""

    def evaluate(self, x: NDArray, level: Level = Level.GRAD) -> BatchPotentialResult:
        if isinstance(self.potential, OnTheFlyPotential):
            self._tracker += 1
            results = []
            for bead, coords in enumerate(x):
                results.append(self._evaluate_onthefly(coords, level, self._tracker, bead))
            return self._stack_results(results)

        func = partial(self.potential, level=level)
        return self._stack_results(tuple(map(func, x)))


class CachedExecutor(Executor):
    """Executor wrapper with per-geometry result caching."""

    def __init__(self, executor: Executor):
        if not isinstance(executor, Executor):
            raise TypeError("CachedExecutor wraps an Executor, not a Potential.")
        self.executor = executor
        self._cached = self._new_cache()

    def clear_cache(self) -> None:
        self._cached = self._new_cache()

    def evaluate(self, x: NDArray, level: Level = Level.GRAD) -> ExecutorResult:
        x = np.asarray(x)
        if x.ndim <= 2:
            return self._evaluate_one(x, level)
        return self._stack_results([self._evaluate_one(coords, level) for coords in x])

    def _evaluate_one(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult:
        cache_key = self._cache_key(x)
        cached = self._get_cached(self._cached, cache_key, level)
        if cached is not None:
            return cached
        result = self._evaluate_uncached(x, level)
        self._store_cached(self._cached, cache_key, result)
        return result

    def _evaluate_uncached(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult:
        return self.executor.evaluate(x, level)

    @staticmethod
    def _new_cache() -> list[dict[int, PotentialResult]]:
        return [dict() for _ in Level]

    @staticmethod
    def _cache_key(x: NDArray) -> int:
        x = np.ascontiguousarray(x)
        return hash((x.shape, x.dtype.str, x.tobytes()))

    @staticmethod
    def _get_cached(cache: list[dict[int, PotentialResult]], cache_key: int, level: Level) -> PotentialResult | None:
        for cached_level in range(int(level), len(cache)):
            result = cache[cached_level].get(cache_key)
            if result is not None:
                energy, grad, hess = result
                return (
                    energy,
                    grad if level >= Level.GRAD else None,
                    hess if level >= Level.FREQ else None,
                )
        return None

    @staticmethod
    def _store_cached(cache: list[dict[int, PotentialResult]], cache_key: int, result: PotentialResult) -> None:
        _, grad, hess = result
        level = Level.FREQ if hess is not None else Level.GRAD if grad is not None else Level.ENER
        cache[level][cache_key] = result


class FixAtom(CachedExecutor):
    """Executor wrapper that appends fixed atoms before evaluation."""

    def __init__(self, executor: Executor, x_fix: NDArray, dx=(None, None)):
        super().__init__(executor)
        self.n_fix = len(x_fix)
        self.x_fix = x_fix
        self.grad_dx, self.hess_dx = dx
        self._cached_augmented: list[dict[int, PotentialResult]] = self._new_cache()

    def clear_cache(self) -> None:
        super().clear_cache()
        self._cached_augmented = self._new_cache()

    def _evaluate_uncached(self, x: NDArray, level: Level = Level.GRAD) -> PotentialResult:
        x_full = np.r_[x, self.x_fix]
        if level == Level.FREQ:
            if self.hess_dx is None:
                res = list(self._evaluate_augmented(x_full, level))
            elif self.grad_dx is None:
                res = list(self._evaluate_augmented(x_full, level - 1))
                res[2] = hess_from_grad(x_full, lambda y: self._evaluate_augmented(y, level - 1)[1], self.hess_dx)
            else:
                res = list(self._evaluate_augmented(x_full, level - 2))
                res[1] = grad_from_energy(x_full, lambda y: self._evaluate_augmented(y, level - 2)[0], self.grad_dx)
                res[2] = hess_from_energy(x_full, lambda y: self._evaluate_augmented(y, level - 2)[0], self.hess_dx)
            res[1] = res[1][: -self.n_fix]
            res[2] = res[2][: -self.n_fix * 3, : -self.n_fix * 3]
        elif level == Level.GRAD:
            if self.grad_dx is None:
                res = list(self._evaluate_augmented(x_full, level))
            else:
                res = list(self._evaluate_augmented(x_full, level - 1))
                res[1] = grad_from_energy(x_full, lambda y: self._evaluate_augmented(y, level - 1)[0], self.grad_dx)
            res[1] = res[1][: -self.n_fix]
        else:
            res = self._evaluate_augmented(x_full, level)
        return tuple(res)

    def _evaluate_augmented(self, x: NDArray, level: Level) -> PotentialResult:
        cache_key = self._cache_key(x)
        cached = self._get_cached(self._cached_augmented, cache_key, level)
        if cached is not None:
            return cached
        result = self.executor.evaluate(np.asarray(x), level)
        self._store_cached(self._cached_augmented, cache_key, result)
        return result


class ParallelExecutor(Executor):
    """Parallel dispatcher for on-the-fly driver processes."""

    def __init__(
        self,
        symbols: list[str],
    ):
        self.symbols = list(symbols)
        self._tracker = -1
        self.timeout_ms = PARALLEL_TIMEOUT_MS
        self.server_info_path = SERVER_INFO_PATH
        self.server_info_tmp_path = self.server_info_path.with_name(f"{self.server_info_path.name}.tmp")

        self._zmq_context = zmq.Context.instance()
        self.socket = self._zmq_context.socket(zmq.ROUTER)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.bind("tcp://*:0")
        self.endpoint = self.socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.active_drivers: set[bytes] = set()
        self.idle_drivers: set[bytes] = set()
        self._write_server_info()

    def evaluate(self, x: NDArray, level: Level = Level.GRAD):
        x = np.asarray(x)
        self._tracker += 1
        if x.ndim <= 2:
            return self._run_tasks([self._make_task(0, x, level)])[0]
        tasks = [self._make_task(i, coords, level) for i, coords in enumerate(x)]
        return self._stack_results(self._run_tasks(tasks))

    def _make_task(self, task_id: int, coords: NDArray, level: Level) -> "Task":
        return Task(
            id=task_id,
            coords=coords,
            level=int(level),
            tracker=self._tracker,
        )

    def _run_tasks(self, tasks: Sequence["Task"]) -> list[PotentialResult]:
        results: dict[int, PotentialResult] = {}
        pending = len(tasks)
        next_task = 0
        poller = zmq.Poller()
        poller.register(self.socket, zmq.POLLIN)

        while pending > 0:
            next_task = self._dispatch_ready(tasks, next_task)
            events = dict(poller.poll(self.timeout_ms))
            if self.socket not in events:
                raise TimeoutError("Timed out waiting for parallel potential driver.")
            identity, data = self.socket.recv_multipart()
            completed = self._handle_message(identity, pickle.loads(data), results)
            pending -= completed

        return [results[task.id] for task in tasks]

    def _dispatch_ready(self, tasks: Sequence["Task"], next_task: int) -> int:
        while self.idle_drivers and next_task < len(tasks):
            identity = self.idle_drivers.pop()
            self.socket.send_multipart([identity, pickle.dumps(tasks[next_task])])
            next_task += 1
        return next_task

    def _handle_message(self, identity: bytes, message: object, results: dict[int, PotentialResult]) -> int:
        if identity not in self.active_drivers:
            self.active_drivers.add(identity)

        if isinstance(message, dict) and message.get("type") == "ready":
            self.idle_drivers.add(identity)
            return 0

        if isinstance(message, Result):
            if message.error is not None:
                raise RuntimeError(message.error)
            results[message.id] = (message.energy, message.grad, message.hess)
            self.idle_drivers.add(identity)
            return 1

        raise TypeError(f"Unexpected driver message: {type(message)}")

    def _write_server_info(self) -> None:
        host = get_bind_addr()
        port = int(self.endpoint.rsplit(":", 1)[1])
        info = {
            "host": host,
            "port": port,
            "symbols": self.symbols,
        }
        with self.server_info_tmp_path.open("w", encoding="utf-8") as f:
            json.dump(info, f)
        os.replace(self.server_info_tmp_path, self.server_info_path)

    def close(self) -> None:
        for driver in self.active_drivers:
            self.socket.send_multipart([driver, pickle.dumps(None)])
        self.socket.close(linger=0)
        with contextlib.suppress(FileNotFoundError):
            self.server_info_path.unlink()
        with contextlib.suppress(FileNotFoundError):
            self.server_info_tmp_path.unlink()


@dataclass(frozen=True)
class Task:
    """executor -> driver"""

    id: int
    coords: NDArray
    level: int
    tracker: int


@dataclass(frozen=True)
class Result:
    """driver -> executor"""

    id: int
    energy: float
    grad: NDArray | None = None
    hess: NDArray | None = None
    error: str | None = None


class Driver:
    """Driver worker that connects to a ParallelExecutor and evaluates tasks."""

    def __init__(
        self,
        potential: OnTheFlyPotential,
        *,
        identity: str | None = None,
    ):
        if not isinstance(potential, OnTheFlyPotential):
            raise TypeError("Parallel driver workers require an OnTheFlyPotential potential.")
        self.potential = potential
        self.identity = identity or get_driver_id()
        self.socket = zmq.Context.instance().socket(zmq.DEALER)
        self.socket.setsockopt(zmq.IDENTITY, self.identity.encode())

    def run(self) -> None:
        endpoint = self._read_endpoint()
        self.socket.connect(endpoint)
        self.socket.send_pyobj({"type": "ready"})
        while True:
            task = self.socket.recv_pyobj()
            if task is None:
                break
            self.socket.send_pyobj(self._run_task(task))

    def close(self) -> None:
        self.socket.close(linger=0)

    def _run_task(self, task: Task) -> Result:
        try:
            return self._evaluate_task(task)
        except Exception as exc:
            logger.exception("Driver task %s failed", task.id)
            return Result(task.id, 0.0, error=str(exc))

    def _evaluate_task(self, task: Task) -> Result:
        self.potential._folder = Executor._prepare_workdir(
            self.potential._working_dir,
            task.tracker,
            task.id,
        )
        energy, grad, hess = self.potential(task.coords, Level(task.level))
        return Result(task.id, energy, grad, hess)

    def _read_endpoint(self) -> str:
        info = read_server_info()
        return f"tcp://{info['host']}:{int(info['port'])}"


def get_bind_addr() -> str:
    slurmd_nodename = os.environ.get("SLURMD_NODENAME")
    if slurmd_nodename:
        return slurmd_nodename
    for env_name in ("SLURM_NODELIST", "SLURM_JOB_NODELIST"):
        nodelist = os.environ.get(env_name)
        if nodelist:
            return nodelist.split(",", 1)[0].split("[", 1)[0]
    return socket.gethostname()


def get_driver_id() -> str:
    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = os.environ.get("SLURM_JOB_ID")
    proc_id = os.environ.get("SLURM_PROCID")

    if array_job_id and array_task_id:
        return f"driver-{array_job_id}_{array_task_id}"
    if job_id and proc_id:
        return f"driver-{job_id}_{proc_id}"
    if job_id:
        return f"driver-{job_id}"
    return f"driver-{os.getpid()}"
