#!/usr/bin/env bash

set -e

export PYTHONPATH=.:$PYTHONPATH

main.py H_guess.xyz -o H -T 200 --mode min --phase gas -P custom_pes:CustomPES -F H.json
main.py H2_guess.xyz -o H2 -T 200 --mode min --phase gas -P custom_pes:CustomPES -F H2.json --opt lBFGS
main.py TS_guess.xyz -o ts -T 200 --mode ts --phase gas -P custom_pes:CustomPES -F full.json -l H.pkl H2.pkl -p
echo "**** Result should be   1.9234e-20 cm^3 / s"
echo
main.py ts.pkl -o inst-32 -T 300 --mode inst --phase gas -P custom_pes:CustomPES -F full.json -l H.pkl H2.pkl --opt SBW -p --maxiter 20 -N 32 -s 0.1
main.py inst-32.pkl -o inst-32 -T 200 --mode inst --phase gas -P custom_pes:CustomPES -F full.json --opt SBW -p --maxiter 20 -N 32
main.py inst-32.pkl -o inst-64 -T 200 --mode inst --phase gas -P custom_pes:CustomPES -F full.json --opt SBW -p --maxiter 20 -N 64
echo "**** Factor should be about 60"
echo "**** Rate should be 1.048673e-18 cm^3 / s"
echo
