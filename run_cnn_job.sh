#!/bin/bash
#$ -N train_simplecnn
#$ -cwd
#$ -j y
#$ -o output.log
#$ -l gpus=1
#$ -l gpu_type=P100
#$ -l h_rt=02:00:00
#$ -m bea           
#$ -M atuladas@bu.edu  

source ~/.bashrc
conda activate dl4ds

python starter_code.py
