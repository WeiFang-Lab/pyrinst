import pickle
from pathlib import Path

import matplotlib
import numpy as np

from pyrinst.geometries import Instanton

matplotlib.use("Agg")

from pyrinst.cli import plot


def make_test_instanton() -> Instanton:
    coords = np.array(
        [
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
            [[0.0, 0.0, 0.1], [0.0, 0.0, 0.9]],
            [[0.0, 0.0, 0.2], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    geom = Instanton(coords, symbols=["H", "H"], beta=100.0)
    geom.energy = np.array([0.0, 0.1, 0.0], dtype=float)
    return geom


def test_instplot_main_saves_figure_and_data(tmp_path) -> None:
    geom = make_test_instanton()
    input_file = Path(tmp_path) / "inst.pkl"
    output_file = Path(tmp_path) / "instplot.png"

    with input_file.open("wb") as f:
        pickle.dump(geom, f)

    plot.main([str(input_file), "--savefig", str(output_file), "--savedata"])

    assert output_file.exists()
    data_file = input_file.with_suffix(".dat")
    assert data_file.exists()

    data = np.loadtxt(data_file)
    assert data.shape == (3, 2)
    assert np.allclose(data[:, 1], geom.energy)
