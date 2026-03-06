import argparse
import pickle

import numpy as np

from pyrinst.config.constants import KB
from pyrinst.utils.fep import *
from pyrinst.utils.units import Energy


def main():
    parser = argparse.ArgumentParser(description="Generate distribution via quasi random number.")
    parser.add_argument("input", type=str, help="pkl file.")
    parser.add_argument("--prefix", type=str, default="simulation.pos", help="prefix of beads filename")
    parser.add_argument("-n", "--nbeads", type=int, default=24, help="The number of beads.")
    args = parser.parse_args()

    # load beads energies and harm energies
    with open(args.input, "rb") as f:
        input_geom = pickle.load(f)

    # beads_energies = np.loadtxt('beads_energies.txt').T
    beads_energies = extract_energy_from_xyz(prefix=args.prefix, nbeads=args.n, is_instanton=args.inst).T

    beads_energies = beads_energies[1:] - beads_energies[0]
    aes = np.average(beads_energies, axis=1)
    # bhs = np.loadtxt('harm_energies.txt').T
    bhs = input_geom.harm_energies.T
    des = (aes - bhs) * Energy(1, "eV").get("Hartree")  # eV -> Hartree

    beta = 1.0 / (input_geom.T * KB)  # 1/Hartree

    freqs_complex = to_complex(input_geom.freqs)
    df0 = dF(freqs_complex, beta, 24)
    df1, var1 = free_energy_perturbation(des, beta, 0)
    print("ref:", df0.real * Energy(1, "Hartree").get("eV"), "eV")
    print("correction:", df1 * Energy(1, "Hartree").get("eV"), "eV")
    print(
        f"delta F({input_geom.T}K)=: {df0.real * Energy(1, 'Hartree').get('eV') + df1 * Energy(1, 'Hartree').get('eV'):.10e} eV"
    )
    print("error: ", var1 * Energy(1, "Hartree").get("eV"), "eV")


if __name__ == "__main__":
    main()
