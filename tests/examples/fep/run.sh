#!/usr/bin/env bash

pyrinst-gen-ref C2H2.xyz -P MACE --model_path MACE-OFF24_medium.model
pyrinst-sampling ref.pkl -T 250 -N 8192
./eval.sh
pyrinst-fep-eval ref.pkl --prefix simulation.pos
