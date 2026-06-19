#!/usr/bin/env bash

pyrinst geom water.xyz -P MACE --model-path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model --mode single
pyrinst geom ref.pkl -o inst.pkl -T 300 --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model -N 24 -s 0.189
pyrinst sample inst.pkl -T 300 -N 2048
./eval.sh
pyrinst fep-eval inst.pkl --prefix simulation.pos
