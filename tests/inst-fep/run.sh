#!/usr/bin/env bash

gen_ref.py water.xyz -P MACE --model_path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model
optimize.py ref.pkl -o inst.pkl --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model