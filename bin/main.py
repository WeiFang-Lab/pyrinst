#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
from functools import partial

import numpy as np

from easy_instanton.core.modes import Minimum
from easy_instanton.core.opt import optimizers
from easy_instanton.utils.logging_config import setup_logging

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Initial guess for the optimization in xyz, txt, or pkl format.')
parser.add_argument('-o', '--output', default='opt_geom', help='Final optimized geometry.')
parser.add_argument('-v', '--verbose', type=bool, help='Verbosity level.')
# todo: case-insensitive
parser.add_argument('--phase', choices=('gas', 'liquid', 'solid'), default='gas', help='Phase of the system')
parser.add_argument('-P', '--PES', choices=('custom',), help='Potential energy surface')
# todo: choices
parser.add_argument(
    '-F', '--mainInputFile', help='Main input file (for a SCF calculation): vasp: INCAR; gaussian/orca/molpro/cp2k: '
    'full input file without the geometry. This is also used by CustomPes as the input in dictionary format.')
parser.add_argument('--opt', choices=['EF', 'MEP', 'lBFGS'], default='EF', help='Optimization algorithm to use.')
parser.add_argument('-g', '--gtol', default=1e-3, type=float, help='Tolerance in gradient for optimization.')
parser.add_argument('--maxstep', default=0.3, type=float, help='Max-step in optimization.')
parser.add_argument('--maxiter', default=10, type=int, help='Max-iters in optimization.')
args = parser.parse_args()

from custom_pes import CustomPes
pes = CustomPes()

prefix, ext = os.path.splitext(args.input)
setup_logging(verbose=args.verbose, log_file=f'{prefix}.log', result_file=f'{prefix}.out')
log = logging.getLogger()

# read input file
if ext == '.xyz':
    raise NotImplementedError  # todo: xyz module
elif ext == '.txt':
    x = np.loadtxt(args.input)  # todo: 1d instanton
    data = Minimum(x, pes=pes)
elif ext == '.pkl':
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
else:
    msg = f'Unknown file format: {args.input}'
    log.error(msg)
    raise ValueError(msg)

prefix, ext = os.path.splitext(args.output)
if ext in {'.xyz', '.txt', '.pkl'}:
    args.output = prefix

opt = optimizers.ModeFollowing(data.order, maxstep=args.maxstep, verbosity=args.verbosity)
opt.search(data, gtol=args.gtol, maxiter=args.maxiter, callback=partial(type(data).output, prefix=args.output))

data.final_output(args.output)
