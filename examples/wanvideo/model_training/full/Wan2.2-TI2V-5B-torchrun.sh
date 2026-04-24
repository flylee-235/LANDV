#!/bin/bash

# 设置工作目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# 🔥🔥🔥 TI2V-5B + 结构编码器训练模式
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    examples/wanvideo/model_training/train_torchrun.py \
    --dataset_base_path /home/610-ltf/DL/Datasets/data \
    --dataset_metadata_path /home/610-ltf/DL/Datasets/data/metadata.csv \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "/home/public/Models/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,/home/public/Models/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,/home/public/Models/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 1 \
    --remove_prefix_in_ckpt "pipe.structure_encoder." \
    --output_path "/home/610-ltf/DL/models/train/ti2v_structure_encoder" \
    --trainable_models "structure_encoder" \
    --max_timestep_boundary 0.417 \
    --min_timestep_boundary 0 \
    --gradient_accumulation_steps 4 \
    --dataset_num_workers 2 \
    --weight_decay 1e-2 \
    --save_steps 2500 \
    --max_grad_norm 1.0 \
    --use_structure_encoder \
    --structure_encoder_dim 768 \
    --extra_inputs "input_image" \
    --data_file_keys "video,input_image,sketch,seg"\
    --resume_from_checkpoint "/home/610-ltf/DL/models/train/ti2v_structure_encoder/step-10000.safetensors"


torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    examples/wanvideo/model_training/train_torchrun.py \
    --dataset_base_path /home/610-ltf/DL/Datasets/data \
    --dataset_metadata_path /home/610-ltf/DL/Datasets/data/metadata.csv \
    --height 480 --width 832 --num_frames 49 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "/home/public/Models/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,/home/public/Models/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,/home/public/Models/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
    --use_structure_encoder \
    --structure_encoder_dim 768 \
    --train_moe \
    --structure_encoder_ckpt "/home/610-ltf/DL/models/train/ti2v_structure_encoder_phase1/latest.safetensors" \
    --extra_inputs "input_image" \
    --data_file_keys "video,input_image,sketch,seg" \
    --learning_rate 5e-5 \
    --num_epochs 1 \
    --output_path "/home/610-ltf/DL/models/train/ti2v_moe_injection_phase2" \
    --trainable_models "dit" \
    --gradient_accumulation_steps 4 \
    --dataset_num_workers 2