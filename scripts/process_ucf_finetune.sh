#!/bin/bash

module purge
module load 2022
module load Miniconda3/4.12.0
module load CUDA/11.7.0

# Activate your environment
source activate llamavid

# create questions.json
python process_ucf_crime_finetune.py
