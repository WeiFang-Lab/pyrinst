#!/usr/bin/env bash

pyrinst geom C2H2.xyz -P MACE --model-path MACE-OFF24_medium.model --mode single
pyrinst sample ref.pkl -T 250 -N 8192
./eval.sh
pyrinst fep-eval ref.pkl --prefix simulation.pos
