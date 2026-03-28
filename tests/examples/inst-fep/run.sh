#!/usr/bin/env bash

gen_ref.py water.xyz -P MACE --model_path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model
optimize.py ref.pkl -o inst.pkl -T 300 --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model -N 24 -s 0.189
sampling.py inst.pkl -T 300 -N 2048
./eval.sh
fep_eval.py inst.pkl --prefix simulation.pos
