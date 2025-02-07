#!/bin/sh
#SBATCH --job-name="coref"
#SBATCH --time=50:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=fastgpus
#SBATCH --output=test.txt

sh run.sh