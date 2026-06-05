import os
from types import SimpleNamespace

import numpy as np
import pytest

from pyrinst.potentials import CachedExecutor, FixAtom, Level, Potential, SingleExecutor
from pyrinst.potentials.base import OnTheFlyPotential


class CountingPotential(Potential):
    type_alias = "_counting_test_potential"

    def __init__(self):
        self.calls: list[tuple[np.ndarray, Level]] = []

    def __call__(self, x, level: Level = Level.GRAD):
        coords = np.array(x, copy=True)
        self.calls.append((coords, level))
        energy = float(np.sum(coords**2))
        grad = 2 * coords if level >= Level.GRAD else None
        hess = np.eye(coords.size) if level >= Level.FREQ else None
        return energy, grad, hess


class FolderRecorderDriver(OnTheFlyPotential):
    type_alias = "_folder_recorder_test_driver"
    _runcmd = "folder-recorder"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.folders = []
        self.cwd_seen = []

    def generate_input(self, x, level: Level = Level.GRAD):
        self.folders.append(self._folder)
        self.cwd_seen.append(os.getcwd())
        with open(os.path.join(self._folder, "input.txt"), "w") as f:
            f.write("input")

    def run(self):
        self.cwd_seen.append(os.getcwd())

    def parse_output(self):
        return SimpleNamespace(energy=1.0, grad=np.zeros((1, 3)), hess=None)


def test_cached_executor_batch_deduplicates_inputs():
    potential = CountingPotential()
    executor = CachedExecutor(SingleExecutor(potential))
    x1 = np.array([[0.0, 0.0, 0.0]])
    x2 = np.array([[0.5, 0.0, 0.0]])
    xs = np.array([x1, x1.copy(), x2])

    energy, grad, hess = executor.evaluate(xs, Level.GRAD)

    np.testing.assert_allclose(energy, [0.0, 0.0, 0.25])
    np.testing.assert_allclose(grad, 2 * xs)
    assert hess is None
    assert len(potential.calls) == 2


def test_fixatom_keeps_subclass_single_point_logic_through_cache():
    potential = CountingPotential()
    fixed = np.array([[0.0, 0.0, 1.0]])
    executor = FixAtom(CachedExecutor(SingleExecutor(potential)), fixed)
    x = np.array([[1.0, 0.0, 0.0]])

    energy, grad, hess = executor.evaluate(x, Level.GRAD)

    assert energy == 2.0
    np.testing.assert_allclose(grad, 2 * x)
    assert hess is None
    assert len(potential.calls) == 1

    executor.evaluate(x.copy(), Level.GRAD)
    assert len(potential.calls) == 1


def test_cached_executor_requires_executor():
    with pytest.raises(TypeError, match="wraps an Executor"):
        CachedExecutor(CountingPotential())


def test_single_executor_assigns_onthefly_folder_without_chdir(tmp_path):
    template = tmp_path / "template.inp"
    template.write_text("template")
    working_dir = tmp_path / "work"
    potential = FolderRecorderDriver(["H"], template_input=str(template), working_dir=str(working_dir))
    executor = SingleExecutor(potential, working_dir=str(working_dir))
    cwd = os.getcwd()

    energy, grad, hess = executor.evaluate(np.zeros((1, 3)), Level.GRAD)

    assert energy == 1.0
    np.testing.assert_allclose(grad, np.zeros((1, 3)))
    assert hess is None
    assert os.getcwd() == cwd
    assert potential.cwd_seen == [cwd, cwd]
    assert potential.folders == [str(working_dir / "0")]
    assert (working_dir / "0" / "input.txt").is_file()
