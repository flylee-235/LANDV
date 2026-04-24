# Wan2.2-T2V-A14B-torchrun.sh - 单卡启动脚本
# Wan2.2-T2V-A14B-torchrun-single-gpu.sh - 单卡启动脚本（修正路径版本）
#!/bin/bash

# 设置工作目录为仓库根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$ROOT_DIR"

# 单卡训练 - High Noise Model
python examples/wanvideo/model_training/train_torchrun_single.py \
  --dataset_base_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset \
  --dataset_metadata_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset/metadata.csv \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "/home/public/Models/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,/home/public/Models/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,/home/public/Models/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "/home/610-ltf/DL/models/train/Wan2.2-T2V-A14B_high_noise_full" \
  --trainable_models "dit" \
  --max_timestep_boundary 0.417 \
  --min_timestep_boundary 0 \
  --gradient_accumulation_steps 8 \
  --dataset_num_workers 4 \
  --weight_decay 1e-2 \
  --save_steps 500 \
  --max_grad_norm 1.0

# Wan2.2-T2V-A14B-torchrun-multi-gpu.sh - 多卡启动脚本
#!/bin/bash

# 多GPU训练启动脚本


# Wan2.2-T2V-A14B-torchrun-multi-gpu.sh - 多卡启动脚本（动态GPU分配版）
#!/bin/bash

# 设置工作目录
cd "$ROOT_DIR"

# 动态GPU分配环境 - 不指定CUDA_VISIBLE_DEVICES
# 系统会自动分配可用的GPU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

torchrun \
    --nproc_per_node=6 \
    --master_port=29501 \
    examples/wanvideo/model_training/train_torchrun.py \
    --dataset_base_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset_copy \
    --dataset_metadata_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset_copy/metadata.csv \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "/home/public/Models/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,/home/public/Models/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,/home/public/Models/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate 1e-5 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.dit." \
    --output_path "/home/610-ltf/DL/models/train/Wan2.2-T2V-A14B_high_noise_full_multi_gpu" \
    --trainable_models "dit" \
    --max_timestep_boundary 0.417 \
    --min_timestep_boundary 0 \
    --gradient_accumulation_steps 2 \
    --dataset_num_workers 2 \
    --weight_decay 1e-2 \
    --save_steps 100 \
    --max_grad_norm 1.0 \
    --use_structure_encoder \
    --structure_encoder_dim 768 \
    --data_file_keys "video"


# 🔥🔥🔥 只训练结构编码器模式
torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    examples/wanvideo/model_training/train_torchrun.py \
    --dataset_base_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset_copy \
    --dataset_metadata_path /home/610-ltf/DL/Datasets/video/data/example_video_dataset_copy/metadata.csv \
    --height 480 \
    --width 832 \
    --num_frames 49 \
    --dataset_repeat 100 \
    --model_id_with_origin_paths "/home/public/Models/Wan2.2-T2V-A14B:high_noise_model/diffusion_pytorch_model*.safetensors,/home/public/Models/Wan2.2-T2V-A14B:models_t5_umt5-xxl-enc-bf16.pth,/home/public/Models/Wan2.2-T2V-A14B:Wan2.1_VAE.pth" \
    --learning_rate 1e-4 \
    --num_epochs 2 \
    --remove_prefix_in_ckpt "pipe.structure_encoder." \
    --output_path "/home/610-ltf/DL/models/train/structure_encoder_only" \
    --trainable_models "structure_encoder" \
    --max_timestep_boundary 0.417 \
    --min_timestep_boundary 0 \
    --gradient_accumulation_steps 4 \
    --dataset_num_workers 2 \
    --weight_decay 1e-2 \
    --save_steps 100 \
    --max_grad_norm 1.0 \
    --use_structure_encoder \
    --structure_encoder_dim 768 \
    --data_file_keys "video"