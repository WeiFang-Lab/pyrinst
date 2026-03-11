#!/usr/bin/env python3

import argparse
import pickle

import numpy as np

from pyrinst.geometries import HarmRef, InstRef
from pyrinst.io.xyz import save
from pyrinst.utils.pimc import HarmFEP, InstFEP


def main():
    parser = argparse.ArgumentParser(description="Generate distribution via quasi random number.")
    parser.add_argument("input", help="pkl file.")
    parser.add_argument("-T", type=float, default=300, help="Temperature (K).")
    parser.add_argument("-N", type=int, default=4096, help="The number of configurations sampled.")
    parser.add_argument("-n", "--nbeads", type=int, default=24, help="The number of beads.")
    parser.add_argument(
        "-o", "--output", type=str, default="simulation.pos", help="The prefix of output configuration files."
    )
    parser.add_argument("-l", "--lmd_val", type=float, default=1.0, help="The mass scaling factor.")
    parser.add_argument("--nprandom", action="store_true", help="Use numpy function to generate gaussian samples")
    args = parser.parse_args()

    input_geom = np.load(args.input, allow_pickle=True)
    if type(input_geom) is HarmRef:
        input_geom.T = args.T
        input_geom.N = args.nbeads
        polymer = HarmFEP(input_geom, nbeads=args.nbeads)
    elif type(input_geom) is InstRef:
        if not np.isclose(input_geom.T, args.T):
            raise ValueError(f"Temperature mismatch: {input_geom.T} != {args.T}")
        polymer = InstFEP(input_geom, nbeads=args.nbeads)
    else:
        raise ValueError(f"Invalid geometry type: {type(input_geom)}")

    sampled_nm_pos = polymer.sample_normal_modes(n_samples=args.N)
    sampled_bead_pos = polymer.get_cart_pos(nm_pos=sampled_nm_pos)

    # refresh freqs and add harm_energies
    if type(input_geom) is HarmRef:
        input_geom.freqs = polymer.freqs
    input_geom.harm_energies = polymer.harm_energies
    with open(args.input, "wb") as f:
        pickle.dump(input_geom, f)

    for bead_idx in range(args.nbeads):
        filename = f"{args.output}_{str(bead_idx).zfill(len(str(args.nbeads)))}.xyz"
        # Extract positions for this bead across all samples
        bead_positions = sampled_bead_pos[:, bead_idx, :]  # Shape: (N, natoms*3)

        # Create list of x
        x_list = []
        for sample_idx in range(args.N):
            pos_3d = bead_positions[sample_idx].reshape(-1, 3)  # Shape: (natoms, 3)
            x_list.append(pos_3d)  # Shape: (frames, natoms, 3)

        # Write file
        save(filename, x_list, input_geom.symbols, comment=" ")


if __name__ == "__main__":
    main()
