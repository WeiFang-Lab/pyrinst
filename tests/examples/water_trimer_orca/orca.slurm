#!/bin/bash
#SBATCH --job-name=orca
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.out.err

module load orca/6.1.1
export PYTHONBUFFERED=1

./run.sh
