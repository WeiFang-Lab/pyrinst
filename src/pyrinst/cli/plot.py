import argparse
import itertools
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from pyrinst.geometries import Geometry, Instanton
from pyrinst.io.xyz import load as load_xyz
from pyrinst.utils.coordinates import mass_weight
from pyrinst.utils.elements import element_data
from pyrinst.utils.units import AMU, Energy

ENERGY_COMMENT_PATTERN = r"V\s*=\s*([-+0-9.eE]+)"


@dataclass(slots=True)
class PlotPayload:
    label: str
    coords: NDArray | None
    energies: NDArray | float
    masses: NDArray | None
    symbols: NDArray | None
    geometry: Geometry | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plot the potential along an instanton path against the mass-weighted path length. "
            "Supports pyrinst .pkl files and xyz trajectories."
        )
    )
    parser.add_argument("files", nargs="+", help="xyz or pkl file(s)")
    parser.add_argument("-i", "--index", default=None, help="Text file recording indices of beads to highlight")
    parser.add_argument("-r", "--reverse", action="store_true", help="Reverse the direction of the path")
    parser.add_argument("--centre", action="store_true", help="Highlight the centre bead in the plot")
    parser.add_argument(
        "-u",
        "--units",
        metavar=("FROM", "TO"),
        nargs=2,
        help="Convert energies from unit FROM to unit TO, for example: -u au eV",
    )
    parser.add_argument("--spline", action="store_true", help="Use spline interpolation for the energy curve")
    parser.add_argument("--zpe", action="store_true", help="Plot V + ZPE when bead-wise Hessians are available")
    parser.add_argument("-s", "--shift", nargs="+", type=float, default=(0.0,), help="Shift energies before conversion")
    parser.add_argument("--figsize", nargs=2, type=float, help="Figure size in inches")
    parser.add_argument("--savefig", help="Save plot to file")
    parser.add_argument("--savedata", action="store_true", help="Save plotted r-V data to .dat files")
    parser.add_argument(
        "-ax",
        "--align_x",
        choices=["max", "imax", "center_bead", "half", "l", "r"],
        default="imax",
        help="How to align path coordinates between profiles",
    )
    parser.add_argument(
        "-ay",
        "--align_y",
        choices=["max", "imax", "l", "r"],
        default=None,
        help="How to align energies between profiles",
    )
    parser.add_argument("-m", "--markers", type=str.lower, choices=["normal", "small", "none"], default="normal")
    return parser


def _coerce_energy_array(value: Any) -> NDArray | float:
    if value is None:
        msg = "Input file does not contain energy information."
        raise ValueError(msg)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    return arr


def _parse_xyz_energies(comments: Any) -> NDArray | float:
    if comments is None:
        msg = "XYZ file does not contain readable comment lines."
        raise ValueError(msg)

    if np.isscalar(comments):
        match = re.search(ENERGY_COMMENT_PATTERN, str(comments))
        return np.nan if match is None else float(match.group(1))

    values = []
    for comment in np.asarray(comments).tolist():
        match = re.search(ENERGY_COMMENT_PATTERN, str(comment))
        values.append(np.nan if match is None else float(match.group(1)))
    return np.asarray(values, dtype=float)


def load_payload(filename: str) -> PlotPayload:
    path = Path(filename)
    ext = path.suffix.lower()

    if ext == ".pkl":
        with path.open("rb") as f:
            data = pickle.load(f)
        if not isinstance(data, Geometry):
            msg = f"Unsupported object in {filename}: {type(data)}"
            raise TypeError(msg)
        coords = np.asarray(data.x)
        masses = None if data.m is None else np.asarray(data.m, dtype=float)
        symbols = None if data.symbols is None else np.asarray(data.symbols)
        energies: NDArray | float
        energies = _coerce_energy_array(data.energy) if isinstance(data, Instanton) else _coerce_energy_array(data.V)
        return PlotPayload(path.stem, coords, energies, masses, symbols, data)

    if ext == ".xyz":
        symbols, coords, comments = load_xyz(filename, energy_pattern=True)
        masses = np.asarray(element_data.get_masses(symbols), dtype=float) * AMU
        energies = _parse_xyz_energies(comments)
        return PlotPayload(path.stem, np.asarray(coords), energies, masses, np.asarray(symbols), None)

    msg = f"Unsupported file type: {filename}"
    raise ValueError(msg)


def cumulative_path_length(coords: NDArray, masses: NDArray | None = None) -> NDArray:
    coords = np.asarray(coords, dtype=float)
    if coords.ndim < 2:
        msg = f"Expected a path-like coordinate array, got shape {coords.shape}"
        raise ValueError(msg)

    diffs = np.diff(coords, axis=0)
    if coords.ndim == 3 and masses is not None:
        masses = np.asarray(masses, dtype=float)
        step_sq = np.einsum("a,nad,nad->n", masses, diffs, diffs)
    else:
        step_sq = np.sum(diffs**2, axis=tuple(range(1, diffs.ndim)))
    return np.concatenate(([0.0], np.cumsum(np.sqrt(step_sq))))


def maybe_make_spline(r: NDArray, values: NDArray) -> CubicSpline | None:
    if len(r) < 2 or np.any(np.diff(r) <= 0):
        return None
    try:
        return CubicSpline(r, values)
    except ValueError:
        return None


def apply_energy_units(values: NDArray, units_arg: tuple[str, str] | None) -> tuple[NDArray | float, str | None]:
    if units_arg is None:
        return values, None
    src_unit, dst_unit = units_arg
    converted = Energy(values, src_unit).get(dst_unit)
    return converted, dst_unit


def compute_zpe_profile(geometry: Geometry, masses: NDArray) -> NDArray:
    hess = geometry.hess
    if hess is None:
        msg = "Hessian data is not available in the input object."
        raise ValueError(msg)

    hess = np.asarray(hess, dtype=float)
    if hess.ndim != 3:
        msg = "ZPE plotting requires bead-wise Hessians with shape (n_bead, dof, dof)."
        raise ValueError(msg)

    dim = geometry.x.shape[-1]
    zpe = np.empty(len(hess), dtype=float)
    for i, h_i in enumerate(hess):
        eigvals = np.linalg.eigvalsh(mass_weight(h_i, masses, dim=dim))
        freqs = np.sqrt(np.abs(eigvals)) * np.sign(eigvals)
        zpe[i] = 0.5 * np.sum(freqs[freqs > 0])
    return zpe


def _bn_percentages(geometry: Geometry) -> NDArray | None:
    if not isinstance(geometry, Instanton):
        return None
    dx = np.diff(geometry.x, axis=0)
    bn_atom = geometry.m[:, None] * 2 * np.sum(dx**2, axis=0)
    bn_atom = np.sum(bn_atom, axis=-1)
    total = np.sum(bn_atom)
    if np.isclose(total, 0.0):
        return np.zeros_like(bn_atom)
    return 100.0 * bn_atom / total


def print_path_analysis(payload: PlotPayload) -> None:
    coords = payload.coords
    masses = payload.masses
    if coords is None or coords.ndim != 3 or masses is None:
        return

    dx_cart = np.diff(coords, axis=0)
    dr_cart = np.linalg.norm(dx_cart, axis=2)
    rs_cart = np.sum(dr_cart, axis=0)

    dr_mw = np.sqrt(np.sum((dx_cart**2) * masses[None, :, None], axis=2))
    rs_mw = np.sum(dr_mw, axis=0)

    total_cart = cumulative_path_length(coords)[-1]
    total_mw = cumulative_path_length(coords, masses)[-1]
    bn_pct = None if payload.geometry is None else _bn_percentages(payload.geometry)

    atom_labels = payload.symbols.tolist() if payload.symbols is not None else ["X"] * coords.shape[1]
    print(f"{payload.label}: tunnelling path analysis")
    if bn_pct is None:
        print(f'{"":>14s}   {"path":>9s}  {"path(mw)":>9s}')
    else:
        print(f'{"":>14s}   {"path":>9s}  {"path(mw)":>9s}  {"BN%":>6s}')

    for i, atom in enumerate(atom_labels):
        if bn_pct is None:
            print(f"  atom {i:>2d} ({atom:<3s})  {rs_cart[i]:>9.4f}  {rs_mw[i]:>9.4f}")
        else:
            print(f"  atom {i:>2d} ({atom:<3s})  {rs_cart[i]:>9.4f}  {rs_mw[i]:>9.4f}  {bn_pct[i]:>5.1f}%")

    if bn_pct is None:
        print(f'{"Total":>15s}  {total_cart:>9.4f}  {total_mw:>9.4f}')
    else:
        print(f'{"Total":>15s}  {total_cart:>9.4f}  {total_mw:>9.4f}  {np.sum(bn_pct):>5.1f}%')


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    idx: NDArray[np.integer] | None = None if args.index is None else np.atleast_1d(np.loadtxt(args.index, dtype=int))

    fig, ax = plt.subplots(figsize=args.figsize)
    markers = {
        "normal": itertools.cycle(("o", "^", "v", "s", "p")),
        "small": itertools.cycle((".", "1", "2")),
        "none": itertools.cycle((" ",)),
    }[args.markers]
    colors = itertools.cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    shifts = itertools.cycle(args.shift)

    energy_unit_label: str | None = None

    for filename in args.files:
        payload = load_payload(filename)
        print(filename)

        energies = np.array(payload.energies) if not np.isscalar(payload.energies) else float(payload.energies)
        energies += next(shifts)
        energies, current_unit = apply_energy_units(energies, args.units)
        if current_unit is not None:
            energy_unit_label = current_unit

        if np.isscalar(energies):
            if args.align_y is not None:
                print("Warning: single-point energies are skipped when --align_y is used.")
                continue
            ax.add_artist(ax.axhline(y=energies, color="k", linestyle="dashed", zorder=-10))
            ax.plot([], [], color="k", linestyle="--", label=payload.label)
            continue

        coords = np.asarray(payload.coords)
        if args.reverse:
            coords = coords[::-1]
            energies = energies[::-1]

        r = cumulative_path_length(coords, payload.masses)
        spline = maybe_make_spline(r, energies)
        r_spline = values_spline = None

        x_shift = 0.0
        y_shift = 0.0
        if args.align_x in {"imax", "max"} or args.align_y in {"imax", "max"}:
            if args.align_x == "imax" or args.align_y == "imax":
                x_max = r[np.argmax(energies)]
                y_max = float(np.max(energies))
                if args.align_x == "imax":
                    x_shift = x_max
                if args.align_y == "imax":
                    y_shift = y_max
            if args.align_x == "max":
                x_shift = r[np.argmax(energies)]
            if args.align_y == "max":
                y_shift = float(np.max(energies))
        if args.align_x == "l":
            x_shift = r[0]
        if args.align_y == "l":
            y_shift = float(energies[0])
        if args.align_x == "r":
            x_shift = r[-1]
        if args.align_y == "r":
            y_shift = float(energies[-1])
        if args.align_x == "center_bead":
            centre = len(coords) // 2
            x_shift = r[centre] if len(coords) % 2 else 0.5 * (r[centre] + r[centre - 1])
        if args.align_x == "half":
            x_shift = 0.5 * (r[0] + r[-1])

        r = r - x_shift
        energies = energies - y_shift

        if args.spline and spline is not None:
            r_spline = np.linspace(r[0], r[-1], 1000)
            values_spline = spline(r_spline + x_shift) - y_shift

        current_color = next(colors)
        current_marker = next(markers)
        ax.plot(r, energies, "-", linewidth=0, marker=current_marker, markersize=4, color=current_color, zorder=0)
        if r_spline is not None and values_spline is not None:
            ax.plot(r_spline, values_spline, "-", color=current_color, zorder=0)
        else:
            ax.plot(r, energies, "-", color=current_color, zorder=0)
        ax.plot([], [], label=payload.label, marker=current_marker, markersize=4, color=current_color)

        if idx is not None:
            idx_valid = idx[(idx >= 0) & (idx < len(r))]
            if len(idx_valid):
                ax.scatter(r[idx_valid], energies[idx_valid], marker=current_marker, s=16, color="black", zorder=1)

        if args.centre:
            centre = len(coords) // 2
            ax.plot(r[centre], energies[centre], "o", color=current_color, markersize=6)

        if args.zpe:
            if payload.geometry is None or payload.masses is None:
                raise ValueError("--zpe is only available for pyrinst .pkl files with Hessian data.")
            zpe = compute_zpe_profile(payload.geometry, payload.masses)
            zpe, _ = apply_energy_units(zpe, args.units)
            ax.plot(r, energies + zpe, "--", color=current_color)
            ax.plot([], [], "--", color=current_color, label=f"{payload.label} + ZPE")

        if args.savedata:
            data_path = Path(filename).with_suffix(".dat")
            np.savetxt(data_path, np.column_stack((r, energies)))
            if args.zpe:
                np.savetxt(data_path.with_name(data_path.stem + "_zpe.dat"), np.column_stack((r, zpe)))

        print_path_analysis(
            PlotPayload(
                label=payload.label,
                coords=coords,
                energies=energies,
                masses=payload.masses,
                symbols=payload.symbols,
                geometry=payload.geometry,
            )
        )

    plt.rcParams["pdf.fonttype"] = 42
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(rf"$V \; [\mathrm{{{energy_unit_label}}}]$" if energy_unit_label is not None else r"$V$")
    fig.tight_layout()
    ax.legend(loc="best", prop={"size": 8})

    if args.savefig is not None:
        fig.savefig(args.savefig, dpi=600)
    else:
        plt.show()


if __name__ == "__main__":
    main()
