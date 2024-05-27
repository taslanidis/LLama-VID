#!/bin/bash

module purge
module load 2022
module load Miniconda3/4.12.0
module load CUDA/11.7.0

# Activate your environment
source activate llamavid

# create questions.json
python ./process_ucf_crime_data.py

cp /scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/questions.json /scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/answers.json
cp /scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/test_questions.json /scratch-shared/scur0405/data/LLaMA-VID-Eval/ucf-crime/test_answers.json