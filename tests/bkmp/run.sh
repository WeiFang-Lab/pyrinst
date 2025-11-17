#!/usr/bin/env bash

set -e

export PYTHONPATH=.:$PYTHONPATH

main.py H_guess.xyz -o H -T 200 --mode min --phase gas -F H.json
main.py H2_guess.xyz -o H2 -T 200 --mode min --phase gas -F H2.json --opt lBFGS
main.py TS_guess.xyz -o ts -T 200 --mode ts --phase gas -F full.json -l H.pkl H2.pkl
