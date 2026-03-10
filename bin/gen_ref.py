#!/usr/bin/env python3

import argparse

import numpy as np

from pyrinst.geometries import HarmRef
from pyrinst.io.xyz import load
from pyrinst.potentials import MACE
from pyrinst.utils.coordinates import is_linear
from pyrinst.utils.units import Temperature

HBAR: float = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Centroid structure in xyz format.")
    parser.add_argument("-o", "--output", default="ref", help="filename of reference pkl.")
    parser.add_argument("-P", "--PES", type=str, choices=("MACE",), help="Potential energy surface")
    # parameters for mace calculator
    parser.add_argument("--model_path", help="path of MACE model path", type=str)
    parser.add_argument("--dtype", help="dtype of MACE model", type=str, default="float64")
    parser.add_argument("--device", help="device which model runs on", type=str, default="cuda")
    parser.add_argument("--enable_cueq", action="store_true", help="Enable CUEQ (default: disabled)")

    args = parser.parse_args()

    x, symbols = load(args.input, return_symbols=True)
    if args.PES == "MACE":
        mace_pes = MACE(
            symbols,
            model_paths=args.model_path,
            default_dtype=args.dtype,
            device=args.device,
            enable_cueq=args.enable_cueq,
        )

    reference = HarmRef(x, mace_pes.symbols, n_zero=(5 if is_linear(x) else 6))
    mace_pes.compute(reference, task=2)
    reference.calc_freq()
    reference.norm_dimensionless_modes()
    min_freq: float = min(reference.freqs[np.argpartition(abs(reference.freqs), reference.n_zero)[reference.n_zero :]])
    if min_freq > 0:
        print("All frequencies are real.")
    else:
        print("max imaginary freq:", reference.freqs[0])
        print("crossover T:", Temperature.to_kelvin(2 * np.pi / (HBAR * -min_freq)))
    reference.save(args.output)


if __name__ == "__main__":
    main()
