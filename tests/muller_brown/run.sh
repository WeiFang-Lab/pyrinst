#!/usr/bin/env bash

main.py min2_guess.txt -o min -T 300 --mode min --phase solid
main.py TS1_guess.txt -o ts -T 300 --mode ts --phase solid -l min.pkl
main.py ts.pkl -o inst -T 300 --mode inst -N 128 --maxiter 100 --opt SBW --phase solid
