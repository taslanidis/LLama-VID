#!/bin/bash

#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpumem:80g
#SBATCH --cpus-per-task=18
#SBATCH --time=03:00:00
#SBATCH --job-name=InstallEnvironment
#SBATCH --output=./out/zs_animal_grounding_s1_full-%A.out

set -e

module purge
module load StdEnv
module load eth_proxy
module load cuda/11.7.0

source /cluster/scratch/kfanaras/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate llamavid

CUDA_VISIBLE_DEVICES='0,1'
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llama-vid/llama-vid-7b-full-224-video-fps-1"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamavid/eval/model_activitynet_qa.py \
    --model-path ./work_dirs/$CKPT \
    --video_dir ./data/LLaMA-VID-Eval/animal-kingdom/grounding/videos \
    --gt_file_question ./data/LLaMA-VID-Eval/animal-kingdom/grounding/test_questions.json \
    --gt_file_answers ./data/LLaMA-VID-Eval/animal-kingdom/grounding/test_answers.json \
    --output_dir ./work_dirs/animal_kingdom/grounding_full_zs/$CKPT \
    --output_name pred \
    --num-chunks $CHUNKS \
    --chunk-idx $IDX \
    --conv-mode vicuna_v1 &

done

wait