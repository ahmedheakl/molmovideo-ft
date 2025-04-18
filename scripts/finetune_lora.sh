#!/bin/bash
export PATH=~/local_cuda/bin:$PATH
export LD_LIBRARY_PATH=~/local_cuda/lib64:$LD_LIBRARY_PATH
export PATH="/l/users/salman.khan/conda_envs/.molmo-train/bin:$PATH"
# You can use 2B instead of 7B
MODEL_NAME="allenai/Molmo-7B-D-0924"
# MODEL_NAME="allenai/Molmo-7B-O-0924"

export PYTHONPATH=src:$PYTHONPATH

# ff_out is the lm_head layer in other models.
# I can't find the exact embed_token so, its better to just tune the ff_out too.
# --lora_namespan_exclude "['ff_out']"

# Currently, molmo does not support gradient_checkpointing
# Also it only supports fp32 training

deepspeed src/training/train.py \
    --lora_enable True \
    --use_dora False \
    --lora_rank 128 \
    --lora_alpha 256 \
    --lora_dropout 0.05 \
    --num_lora_modules 50 \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path /l/users/salman.khan/workspace_pointing_lmm/datasets/annotations \
    --image_folder /l/users/salman.khan/molmo/pointing_dataset \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_projector False \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir output/vmolmo_lora128_10mods_v2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --learning_rate 1e-4 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.1 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 1 \
    --dataloader_num_workers 4
