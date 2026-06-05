#!/bin/bash

export PYTHONBUFFERED=1
export GAUSS_SCRDIR=/tmp/GAU-pyrinst
mkdir -p ${GAUSS_SCRDIR}

pyrinst-optimize --parallel -g 3e-3 --maxstep 0.1 --mode inst inst_16.pkl -T 200 -N 32 -o inst_32.xyz > inst_32.out

echo "Your results (N=32): "
tail -n3 inst_32.out
echo "Results (N=32) for reference: "
echo "* S/hbar - beta*Vr = 25.3038"
echo "* kinst = 4.8127e-20 cm^3 / s"
echo "* log10(kinst / (cm^3 s^-1)) = -19.31761"

rm -rf ${GAUSS_SCRDIR}
