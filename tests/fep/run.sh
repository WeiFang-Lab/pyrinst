#!/usr/bin/env bash

gen_ref.py C2H2.xyz -P MACE --model_path MACE-OFF24_medium.model
sampling.py ref.pkl -T 250 -N 8192
./eval.sh
fep_eval.py ref.pkl --prefix simulation.pos_eval
