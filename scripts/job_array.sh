#!/bin/bash
#SBATCH -J run
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH -c 40
#SBATCH --time=4:00:00
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --export=ALL
#SBATCH --output=logs/%x.%j_task_%a.log
#SBATCH --signal=SIGUSR1@90
#SBATCH --array 1-3

source scripts/run.sh