#!/bin/bash
export PATH=~/local_cuda/bin:$PATH
export LD_LIBRARY_PATH=~/local_cuda/lib64:$LD_LIBRARY_PATH
export PATH="$CONDA_PREFIX/bin:$PATH"
# git clone https://github.com/hiyouga/LLaMA-Factory
# cd LLaMA-Factory
# pip3 install -e ".[torch,deepspeed,vllm,bitsandbytes,metrics,liger-kernel]"
# pip3 install ujson decord tensorflow tf-keras wandb natsort
# pip3 install flash-attn --no-build-isolation
ANNOTATIONS_PATH="datasets/annotations"
IMAGES_PATH="molmo/pointing_datasets"
OUTPUT_PATH="output/vidmolmo-vid-2f"
MODEL_NAME="allenai/Molmo-7B-D-0924"

export PYTHONPATH=src:$PYTHONPATH

deepspeed src/training/train.py \
    --deepspeed scripts/zero3_offload.json \
    --model_id $MODEL_NAME \
    --data_path $ANNOTATIONS_PATH \
    --image_folder $IMAGES_PATH \
    --freeze_vision_tower False \
    --freeze_llm False \
    --tune_projector True \
    --bf16 True \
    --fp16 False \
    --disable_flash_attn2 False \
    --output_dir $OUTPUT_PATH \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-5 \
    --projector_lr 1e-5 \
    --vision_lr 2e-6 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --gradient_checkpointing False \
    --report_to wandb \
    --lazy_preprocess True \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 10 \
    --dataloader_num_workers 4 \
    --max_grad_norm 1.0 \
    --adam_epsilon 1e-6 \