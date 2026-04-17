#!/bin/bash
# This script is essentially an on-the-fly instanton calculation using the Orca Quantum Chemistry package.
# The system to be tested is the water trimer at T = 300 K, with ab-initio potentials/gradients evaluated at MP2/cc-pVDZ. 
# Hessians are numerically evaluated by orca.
# The number of beads used is N = 32.
# To test functions in orca.py, run orca-test.py (in this directory)

set -e

module load orca/6.1.1
ORCA=$(which orca)

T=300
N=32

export PYTHONBUFFERED=1

pyrinst-optimize -P orca -F header.txt --working-dir beads -g 3e-4 --maxstep 0.1 -p --mode min min_guess.xyz -o min_opt.xyz --runcmd ${ORCA} --hess-method NumFreq > min_opt.out
pyrinst-optimize -P orca -F header.txt --working-dir beads -g 3e-4 --maxstep 0.1 -p --mode ts ts_guess.xyz -o ts_opt.xyz --runcmd ${ORCA} -l min_opt.pkl --hess-method NumFreq > ts_opt.out
pyrinst-optimize -P orca -F header.txt --working-dir beads -g 3e-4 --maxstep 0.1 -p --mode inst ts_opt.pkl -T $T -N $N -o inst_$N.xyz -s 0.2 -F header.txt --runcmd ${ORCA} --hess-method NumFreq > inst_$N.out

echo "Instanton calculation completed! "
echo ""
echo "The instanton rate k_inst"
tail -n3 inst_$N.out
echo "* Results for reference: "
echo "*   S/hbar - beta*Vr = 38.3848"
echo "*   kinst = 4.1166582417493515e-19 = 0.017018830526551433 / s"
echo "*   log10(kinst / s^-1) = -1.7690702863968741"
echo ""
echo "The barrier height and TST rate"
tail -n3 ts_opt.out
echo "* Results for reference: "
echo "*   barrier = 0.0404743 Eh = 1.10136 eV = 106.265 kJ/mol = 25.398 kcal/mol"
echo "*   kEyring = 3.3529094294703793e-21 = 0.0001386138804827874 / s"
echo "*   log10(kEyring / s^-1) = -3.858193278272231"
