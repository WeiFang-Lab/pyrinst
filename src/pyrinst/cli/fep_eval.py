import argparse
import pickle

import numpy as np

from pyrinst.geometries import HarmRef, InstRef
from pyrinst.io.formats import Formats
from pyrinst.io.xyz import load
from pyrinst.utils.fep import effective_sample_size, free_energy_perturbation
from pyrinst.utils.units import EV, KB


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=str, help="pkl file.")
    parser.add_argument("--prefix", type=str, default="simulation.pos", help="prefix of beads filename")
    parser.add_argument("--nbeads", type=int, default=24, help="The number of beads.")


def add_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("fep-eval", help="Evaluate FEP corrections from sampled bead energies.")
    configure_parser(parser)
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    with open(args.input, "rb") as f:
        input_geom = pickle.load(f)
    if args.nbeads != input_geom.N:
        raise ValueError("FEP bead count must match the sampled reference")
    if isinstance(input_geom, InstRef):
        input_geom = input_geom.to_full_ring()

    df0 = input_geom.delta_free_energy()

    filenames = [f"{args.prefix}_{str(bead_idx).zfill(len(str(args.nbeads)))}.xyz" for bead_idx in range(args.nbeads)]
    energy_pattern: str = r"energy\s*=\s*['\"]?([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)['\"]?"
    if type(input_geom) is HarmRef:
        _, _, beads_energies = load(filenames, read_coords=False, energy_pattern=energy_pattern)
        ref_energy = input_geom.energy
        weights = 1
        print("Harmonic FEP")
    elif type(input_geom) is InstRef:
        _, x, beads_energies = load(filenames, energy_pattern=energy_pattern)
        ref_energy = input_geom.energy[:, None]
        x = x.reshape(args.nbeads, -1, len(input_geom.m), 3).transpose(1, 0, 2, 3)
        dx = np.diff(x, axis=1, append=x[:, 0][:, None, ...])
        x0 = input_geom.x
        dx0 = np.diff(x0, axis=0, append=x0[:1])
        weights = 1 if np.isclose(input_geom.BN, 0) else np.maximum(
            np.einsum("ijkl,jkl,k->i", dx, dx0, input_geom.masses) / input_geom.BN, 0
        )
        print("Instanton FEP")
    beads_energies = beads_energies * EV - ref_energy
    aes = np.average(beads_energies, axis=0)
    bhs = input_geom.harm_energies
    des = aes - bhs
    beta = 1.0 / (input_geom.T * KB)
    df1, var1 = free_energy_perturbation(des, beta, weights=weights)
    ess = effective_sample_size(des, beta, weights=weights)

    print(f"reference: {df0 / EV:{Formats.ENERGY}} eV")
    print(f"correction: {df1 / EV:{Formats.ENERGY}} eV")
    print(f"Delta F({input_geom.T} K): {(df0 + df1) / EV:{Formats.ENERGY}} eV")
    print(f"uncertainty: {var1 / EV:{Formats.ENERGY}} eV")
    print(f"ESS: {ess:.2f} / {len(des)} ({ess / len(des):.2%})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate distribution via quasi random number.")
    configure_parser(parser)
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
