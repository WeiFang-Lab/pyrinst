#!/bin/bash

(
  export CUDA_VISIBLE_DEVICES=0
  for d in $(seq -f "%02g" 0 5); do
      mace_eval_configs \
      --configs="simulation.pos_${d}.xyz" \
      --model="MACE-OFF23_medium_water_train3_run-1020_stagetwo.model" \
      --output="simulation.pos_eval_${d}.xyz" \
      --device='cuda' \
      --batch_size=128
  done
) &

(
  export CUDA_VISIBLE_DEVICES=1
  for k in $(seq -f "%02g" 6 11); do
      mace_eval_configs \
      --configs="simulation.pos_${k}.xyz" \
      --model="MACE-OFF23_medium_water_train3_run-1020_stagetwo.model" \
      --output="simulation.pos_eval_${k}.xyz" \
      --device='cuda' \
      --batch_size=128
  done
) &

(
  export CUDA_VISIBLE_DEVICES=2
  for k in $(seq -f "%02g" 12 17); do
      mace_eval_configs \
      --configs="simulation.pos_${k}.xyz" \
      --model="MACE-OFF23_medium_water_train3_run-1020_stagetwo.model" \
      --output="simulation.pos_eval_${k}.xyz" \
      --device='cuda' \
      --batch_size=128
  done
) &

(
  export CUDA_VISIBLE_DEVICES=3
  for k in $(seq -f "%02g" 18 23); do
      mace_eval_configs \
      --configs="simulation.pos_${k}.xyz" \
      --model="MACE-OFF23_medium_water_train3_run-1020_stagetwo.model" \
      --output="simulation.pos_eval_${k}.xyz" \
      --device='cuda' \
      --batch_size=128
  done
) &

wait