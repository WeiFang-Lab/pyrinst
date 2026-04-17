#!/usr/bin/env bash

pyrinst-gen-ref water.xyz -P MACE --model_path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model
pyrinst-optimize ref.pkl -o inst.pkl -T 300 --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model -N 24 -s 0.189
pyrinst-sampling inst.pkl -T 300 -N 2048
./eval.sh
pyrinst-fep-eval inst.pkl --prefix simulation.pos
