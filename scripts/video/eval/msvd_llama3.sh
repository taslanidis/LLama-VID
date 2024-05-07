#!/bin/bash

CKPT="llama-vid/llama-vid-7b-full-224-video-fps-1"
CHUNKS=4

torchrun --nproc_per_node 1 llamavid/eval/eval_msvd_qa_llama.py \
    --pred_path ./work_dirs/eval_msvd_initial/$CKPT \
    --output_dir ./work_dirs/eval_msvd_initial/$CKPT/results2 \
    --output_json ./work_dirs/eval_msvd_initial/$CKPT/results2.json \
    --num_chunks $CHUNKS \
    --num_tasks 2 \