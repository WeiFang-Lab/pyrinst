#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
from functools import partial

import numpy as np

from pyrinst.config.formats import FORMATS
from pyrinst.core import modes_registry
from pyrinst.core.opt import optimizers
from pyrinst.utils.logging_config import setup_logging
from pyrinst.io.xyz import load

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Initial guess for the optimization in xyz, txt, or pkl format.')
parser.add_argument('-o', '--output', default='opt_geom', help='Final optimized geometry.')
parser.add_argument('-v', '--verbose', type=bool, help='Verbosity level.')
temp_group = parser.add_mutually_exclusive_group()
temp_group.add_argument('-T', '--Temp', type=float, help="Temperature in K")
temp_group.add_argument('-b', '--beta', type=float, help="Inverse temperature in whatever unit system you're using")
# todo: case-insensitive
parser.add_argument(
    '--mode', choices=('min', 'ts'), required=True, help='Optimize input to minimum, transition state or instanton.')
parser.add_argument('--phase', choices=('gas', 'liquid', 'solid'), default='gas', help='Phase of the system')
parser.add_argument('-l', '--link', nargs='?', help='Pass min/TS here when optimizing TS/instanton.')
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

prefix, ext = os.path.splitext(args.output)
if ext in {'.xyz', '.txt', '.pkl'}:
    args.output = prefix
setup_logging(verbose=args.verbose, log_file=f'{prefix}.log', result_file=f'{prefix}.out')
log = logging.getLogger()
prefix, ext = os.path.splitext(args.input)

# read input file
if ext == '.xyz':
    x = load(args.input)
    data = modes_registry[args.mode](x, pes)
elif ext == '.txt':
    x = np.loadtxt(args.input)  # todo: 1d instanton
    data = modes_registry[args.mode](x, pes)
elif ext == '.pkl':
    with open(args.input, 'rb') as f:
        data = pickle.load(f)
else:
    msg = f'Unknown file format: {args.input}'
    log.error(msg)
    raise ValueError(msg)
log.info(data.units)

if args.mode == 'ts' and args.link:
    data.reactant = np.load(args.link, allow_pickle=True)

opt = optimizers.ModeFollowing(data.order, maxstep=args.maxstep)
opt.search(data, gtol=args.gtol, maxiter=args.maxiter, callback=partial(type(data).output, prefix=args.output))

data.final_output(args.output)

if args.Temp is not None:
    temp = args.Temp
    beta = data.units.betaTemp(temp)
elif args.beta is not None:
    beta = args.beta
    temp = data.units.betaTemp(beta)
else:
    log.info('Temperature not specified. Program terminated.')
    exit()

log.info('Computing rate...')
fmt: str = FORMATS["temperature"]
log.info(f'T = {temp:{fmt}} K, 1000/T(K) = {1000/temp:{fmt}}; beta = {beta:{fmt}}')

data.calc_rate(beta)
