#!/usr/bin/env python3

import argparse
import logging
import os
import pickle
from functools import partial

import numpy as np

from pyrinst.config.formats import FORMATS
from pyrinst.core import modes_registry, Minimum, TransitionState, Instanton, optimizers
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
    '--mode', choices=('min', 'ts', 'inst'), required=True,
    help='Optimize input to minimum, transition state or instanton.')
parser.add_argument('--phase', choices=('gas', 'liquid', 'solid'), default='gas', help='Phase of the system')
parser.add_argument('-l', '--link', nargs='?', help='Pass min/TS here when optimizing TS/instanton.')
parser.add_argument('-P', '--PES', choices=('custom',), help='Potential energy surface')
# todo: choices
parser.add_argument(
    '-F', '--mainInputFile', help='Main input file (for a SCF calculation): vasp: INCAR; gaussian/orca/molpro/cp2k: '
    'full input file without the geometry. This is also used by CustomPes as the input in dictionary format.')
parser.add_argument('--opt', choices=optimizers.keys(), default='EF', help='Optimization algorithm to use.')
parser.add_argument('-g', '--gtol', default=1e-3, type=float, help='Tolerance in gradient for optimization.')
parser.add_argument('--maxstep', default=0.3, type=float, help='Max-step in optimization.')
parser.add_argument('--maxiter', default=10, type=int, help='Max-iters in optimization.')
parser.add_argument(
    '--no-update', action='store_true', help="Don't use update formula but recompute Hessian at each step.")
parser.add_argument('-N', '--beads', type=int, help='Number of ring-polymer beads (default chosen from input file).')
parser.add_argument('-s', '--spread', default=0.1, type=float, help="Spread of initial guess.")
args = parser.parse_args()

from custom_pes import CustomPES
pes = CustomPES()

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

# temperature
if args.Temp is not None:
    temp: float | None = args.Temp
    beta: float | None = data.units.betaTemp(temp)
elif args.beta is not None:
    beta = args.beta
    temp = data.units.betaTemp(beta)
elif isinstance(data, Instanton):
    beta = data.beta
    temp = data.units.betaTemp(beta)
else:
    beta = temp = None

if args.mode == 'inst':
    if isinstance(data, TransitionState):
        data = data.spread(args.beads, beta, args.spread)
    if args.beads and args.beads != data.n:
        data.interpolate(args.beads)
log.info(data.units)

if args.link:
    match args.mode:
        case 'ts':
            data.rct = np.load(args.link, allow_pickle=True)
        case 'inst':
            link = np.load(args.link, allow_pickle=True)
            if isinstance(link, TransitionState):
                data.ts = link
            elif isinstance(link, Minimum):
                data.rct = link

opt = optimizers[args.opt](order=data.order, maxstep=args.maxstep)
opt.search(data, gtol=args.gtol, maxiter=args.maxiter, callback=partial(type(data).output, prefix=args.output))

data.final_output(args.output)

if beta is None:
    log.info('Temperature not specified. Program terminated.')
    exit()

log.info('\nComputing rate...')
fmt: str = FORMATS["temperature"]
log.info(f'T = {temp:{fmt}} K, 1000/T(K) = {1000/temp:{fmt}}; beta = {beta:{fmt}}')

data.calc_rate(beta)
