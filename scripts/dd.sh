#!/bin/bash
cd /scratch-shared/scur0405/data/LLaMA-VID-Eval
mkdir -p animal_dataset

module purge
module load 2022
module load Miniconda3/4.12.0
module load CUDA/11.7.0

# Activate your environment
source activate llamavid

pip install gdown

gdown https://drive.google.com/uc?id=1X4rL5ey7M1_YM4GDa1DvvVdHoUfuHeJp