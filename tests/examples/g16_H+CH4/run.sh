#!/bin/bash
# Example for H + CH4 using Gaussian
# the b3lyp functional is used with the 6-31G* basis set
# first compute N=16 instanton at T=200
# then reoptimize with N=32 at T=200
T=200
# starting from guess reactant and TS geometries

set -e

export PYTHONBUFFERED=1
module load gaussian/g16c02
export GAUSS_SCRDIR=/tmp/GAU-zyye
mkdir -p ${GAUSS_SCRDIR}

echo "First optimize the stationary points"
optimize.py -P Gaussian --working-dir beads -g 3e-3 --maxstep 0.1 H.xyz --mode min -o Hopt.xyz -F header_doublet --runcmd g16 > H.out
optimize.py -P Gaussian --working-dir beads -g 3e-3 --maxstep 0.1 CH4.xyz --mode min -o CH4opt.xyz -F header_singlet --runcmd g16 > CH4.out
optimize.py -P Gaussian --working-dir beads -g 3e-3 --maxstep 0.1 TS.xyz --mode ts -o TSopt.xyz -l Hopt.pkl CH4opt.pkl -F header_doublet --runcmd g16 > TS.out

echo "Second calculate the instantons for N=16 and N=32"
echo "  We generate the instanton initial guess by spreading around the TS"
optimize.py -P Gaussian --working-dir beads -g 3e-3 --maxstep 0.1 --mode inst TSopt.pkl -T $T -N 16 -o inst_16.xyz -s 0.2 -F header_doublet --runcmd g16 > inst_16.out
optimize.py -P Gaussian --working-dir beads -g 3e-3 --maxstep 0.1 --mode inst inst_16.pkl -T $T -N 32 -o inst_32.xyz -F header_doublet --runcmd g16 > inst_32.out

echo "Your results (N=32): "
tail -n3 inst_32.out
echo "Results (N=32) for reference: "
echo "* S/hbar - beta*Vr = 25.3038"
echo "* kinst = 4.8127e-20 cm^3 / s"
echo "* log10(kinst / (cm^3 s^-1)) = -19.31761"

rm -rf ${GAUSS_SCRDIR}

