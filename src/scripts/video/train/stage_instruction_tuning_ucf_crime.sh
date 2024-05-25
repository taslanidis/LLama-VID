#!/bin/bash

deepspeed llamavid/train/train_mem.py \
    --deepspeed ./scripts/zero2_offload.json \
    --model_name_or_path ./work_dirs/llama-vid/llama-vid-7b-full-224-video-fps-1 \
    --version imgsp_v1 \
    --data_path ./data/LLaMA-VID-Finetune/ucf-crime/crime_detection_qa_ft.json\
    --image_folder ./data/LLaMA-VID-Finetune \
    --video_folder ./data/LLaMA-VID-Finetune \
    --vision_tower ./model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor ./llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --video_fps 1 \
    --video_token 2 \
    --bert_type "qformer_pretrain_freeze_all" \
    --num_query 32 \
    --compress_type "mean" \
    --bf16 True \
    --output_dir ./work_dirs/llama-vid-7b-full-224-video-fps-1-big-ucf-finetuning-from-scratch \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 65536 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to wandb