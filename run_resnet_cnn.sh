#!/bin/bash
#$ -N train_resnet_1
#$ -cwd
#$ -j y
#$ -o output_$JOB_ID.log
#$ -l gpus=1
#$ -l gpu_type=V100
#$ -l h_rt=02:00:00
#$ -m bea           
#$ -M atuladas@bu.edu  

source ~/.bashrc

python Resnet_CNN.py