#!/bin/bash

#SBATCH -J Async_EC
#SBATCH -o Async_EC-%j.out
#SBATCH -e Async_EC-%j.err
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mail-type=ALL
#SBATCH --account=TUK-Amlafoc
#SBATCH --gres=gpu:V100:1


python ./../async_EC.py

echo "Executing on $HOSTNAME"

sleep 5
