import json
import time
from multiprocessing import Process
from types import SimpleNamespace

import numpy as np

from pyrinst.geometries import Geometry
from pyrinst.potentials.base import Level, OnTheFlyPotential
from pyrinst.potentials.executors import Driver, ParallelExecutor


class MockRemoteDriver(OnTheFlyPotential):
    type_alias = "mockremote"
    _runcmd = "mockremote"

    def __init__(self, *args, delay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.delay = delay
        self._last_x = None
        self._last_task = None

    def generate_input(self, x, level: Level = Level.GRAD):
        self._last_x = np.array(x, copy=True)
        self._last_task = level

    def run(self) -> None:
        if self.delay:
            time.sleep(self.delay)

    def parse_output(self):
        grad = 2 * self._last_x if self._last_task >= Level.GRAD else None
        hess = np.eye(self._last_x.size) if self._last_task == Level.FREQ else None
        return SimpleNamespace(
            coord=self._last_x,
            energy=float(np.sum(self._last_x**2)),
            grad=grad,
            hess=hess,
        )


def run_driver(symbols: list[str], template: str, working_dir: str, delay: float, identity: str):
    driver = Driver(
        MockRemoteDriver(symbols, template_input=template, working_dir=working_dir, delay=delay),
        identity=identity,
    )
    try:
        driver.run()
    finally:
        driver.close()


def start_drivers(
    *,
    workers: int = 2,
    symbols: list[str],
    template: str,
    working_dir: str,
    delay: float = 0.0,
):
    processes = [
        Process(
            target=run_driver,
            args=(symbols, template, working_dir, delay, f"test-driver-{i}"),
            daemon=True,
        )
        for i in range(workers)
    ]
    for process in processes:
        process.start()
    return processes


def test_remote_driver_roundtrip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    template = tmp_path / "template.inp"
    template.write_text("template")
    executor = ParallelExecutor(["H", "H"])
    with (tmp_path / "server_info.json").open(encoding="utf-8") as f:
        server_info = json.load(f)
    assert set(server_info) == {"host", "port", "symbols"}
    assert server_info["symbols"] == ["H", "H"]
    processes = start_drivers(
        workers=1,
        symbols=["H", "H"],
        template=str(template),
        working_dir=str(tmp_path / "work"),
    )

    try:
        x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]])
        energy, grad, hess = executor.evaluate(x, Level.GRAD)

        assert energy == np.sum(x**2)
        np.testing.assert_allclose(grad, 2 * x)
        assert hess is None
        assert (tmp_path / "work" / "0" / "0").is_dir()
    finally:
        executor.close()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)


def test_remote_driver_batch_compute_parallel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    template = tmp_path / "template.inp"
    template.write_text("template")
    executor = ParallelExecutor(["H"])
    processes = start_drivers(
        workers=2,
        symbols=["H"],
        template=str(template),
        working_dir=str(tmp_path / "parallel"),
        delay=0.25,
    )

    try:
        time.sleep(0.8)
        geom = Geometry(
            np.array(
                [
                    [[0.0, 0.0, 0.0]],
                    [[0.1, 0.0, 0.0]],
                    [[0.2, 0.0, 0.0]],
                    [[0.3, 0.0, 0.0]],
                ]
            ),
            symbols=["H"],
        )

        start = time.perf_counter()
        executor.compute(geom, Level.GRAD)
        elapsed = time.perf_counter() - start

        np.testing.assert_allclose(geom.V, np.sum(geom.x**2, axis=(1, 2)))
        np.testing.assert_allclose(geom.G, 2 * geom.x)
        assert elapsed < 0.9

        for bead in range(len(geom.x)):
            assert (tmp_path / "parallel" / "0" / str(bead)).is_dir()

        executor.compute(geom, Level.GRAD)
        for bead in range(len(geom.x)):
            assert (tmp_path / "parallel" / "1" / str(bead)).is_dir()
    finally:
        executor.close()
        for process in processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
