#!/bin/bash

### ** Finetune an IP-adapter model for a specific usecase ** ###
# data.json => list of dict: [{"image_file": "1.png", "text": "A dog"}]
# data_root_path => "data/images"
# VRAM requirements: for training with 1024 image resolution requires roughly 30GB, following script works with this!

# 01. Create required format using hf dataset
export HF_TOKEN=""
export HF_DATASET_NAME=""

python3 create_data_from_hf_dataset.py $HF_TOKEN $HF_DATASET_NAME

mkdir checkpoints
pushd checkpoints
    wget https://huggingface.co/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.bin
popd


# 02. Start training
accelerate launch --num_processes 1 --mixed_precision "fp16" \
  tutorial_train_sdxl_plus.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_ip_adapter_path="./checkpoints/ip-adapter-plus_sdxl_vit-h.bin" \
  --image_encoder_path="laion/CLIP-ViT-H-14-laion2B-s32B-b79K" \
  --data_json_file="data.json" \
  --data_root_path="data/images" \
  --mixed_precision="fp16" \
  --resolution=1024 \
  --train_batch_size=1 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-05 \
  --weight_decay=0.01 \
  --output_dir="output_dir" \
  --num_train_epochs=10 \
  --save_steps=2000 \
  --checkpoints_total_limit=1 \
  --push_to_hub \
  --hub_model_id="ip-adapter-plus_sdxl_test"