#!/bin/bash

#SBATCH --job-name="dist-diff"
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=test.txt

ref_path="/export/fs06/kramesh3/"
base_path="/export/fs06/kramesh3/.csv"

python analyze.py --ref_data_path $ref_path \
                  --base_data_path $base_path \