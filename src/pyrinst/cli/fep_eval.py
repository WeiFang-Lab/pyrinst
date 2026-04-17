import argparse
import pickle

import numpy as np

from pyrinst.geometries import HarmRef, InstRef
from pyrinst.io.formats import Formats
from pyrinst.io.xyz import load
from pyrinst.utils.fep import free_energy_perturbation
from pyrinst.utils.units import EV, KB


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate distribution via quasi random number.")
    parser.add_argument("input", type=str, help="pkl file.")
    parser.add_argument("--prefix", type=str, default="simulation.pos", help="prefix of beads filename")
    parser.add_argument("-n", "--nbeads", type=int, default=24, help="The number of beads.")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        input_geom = pickle.load(f)

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
        ref_energy = np.r_[(input_geom.energy, input_geom.energy[::-1])][:, None]
        x = x.transpose(1, 0, 2, 3)
        dx = np.diff(x, axis=1, append=x[:, 0][:, None, ...])
        dx0 = np.diff(input_geom.x, axis=0)
        dx0 = np.concat((dx0, np.zeros_like(dx0[:1]), -dx0[::-1], np.zeros_like(dx0[:1])))
        weights = np.maximum(np.einsum("ijkl,jkl,k->i", dx, dx0, input_geom.masses) / input_geom.BN, 0)
        print("Instanton FEP")
    beads_energies = beads_energies * EV - ref_energy
    aes = np.average(beads_energies, axis=0)
    bhs = input_geom.harm_energies
    des = aes - bhs
    df1, var1 = free_energy_perturbation(des, 1.0 / (input_geom.T * KB), weights=weights)
    print(f"reference: {df0 / EV:{Formats.ENERGY}} eV")
    print(f"correction: {df1 / EV:{Formats.ENERGY}} eV")
    print(f"Delta F({input_geom.T} K): {(df0 + df1) / EV:{Formats.ENERGY}} eV")
    print(f"uncertainty: {var1 / EV:{Formats.ENERGY}} eV")


if __name__ == "__main__":
    main()
