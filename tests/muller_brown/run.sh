#!/usr/bin/env bash

main.py min2_guess.txt -o min -T 300 --mode min --phase model -P mullerbrown --plugin custom_pes.py
main.py TS1_guess.txt -o ts -T 300 --mode ts --phase model -P mullerbrown --plugin custom_pes.py -l min.pkl
main.py ts.pkl -o inst -T 300 --mode inst -N 128 --maxiter 100 --opt SBW --phase model -P mullerbrown --plugin custom_pes.py -s 0.189 --no-update
