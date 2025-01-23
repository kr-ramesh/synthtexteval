#!/bin/bash
#SBATCH --job-name="train-classifier"
#SBATCH --time=48:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --output=temp/testing-classification.txt

echo "hello there"

source ../../.env/bin/activate

sh train.sh bert-base-uncased stanfordnlp/imdb temp/bert-base-uncased-imdb-v1 2 True False
