# train_torchrun.py v26 - 8-bit Adam优化器版本
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import json
import glob
import time
from tqdm import tqdm
# from diffsynth import load_state_dict
# from diffsynth.models import load_state_dict
from diffsynth.models.model_manager import load_state_dict
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from diffsynth.trainers.utils import DiffusionTrainingModule, ModelLogger, wan_parser
from diffsynth.trainers.unified_dataset import UnifiedDataset, LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath, LoadImage
# train_torchrun.py 顶部添加
from examples.wanvideo.dual_cond_encoder import WanAdaptedDualCondEncoder, integrate_structure_encoder_to_pipeline

from examples.wanvideo.moe_attention import inject_moe_into_dit

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ⭐ 检测8-bit优化器
try:
    import bitsandbytes as bnb
    USE_8BIT_ADAM = True
    print("[INFO] bitsandbytes available - will use 8-bit Adam")
except ImportError:
    USE_8BIT_ADAM = False
    print("[WARNING] bitsandbytes not available, using standard AdamW")


class WanTrainingModule(DiffusionTrainingModule):
    def __init__(
        self,
        model_paths=None, model_id_with_origin_paths=None, audio_processor_config=None,
        tokenizer_path=None,
        trainable_models=None,
        lora_base_model=None, lora_target_modules="q,k,v,o,ffn.0,ffn.2", lora_rank=32, lora_checkpoint=None,
        use_gradient_checkpointing=True,
        use_gradient_checkpointing_offload=False,
        extra_inputs=None,
        max_timestep_boundary=1.0,
        min_timestep_boundary=0.0,
        # 🆕 添加结构编码器参数
        use_structure_encoder=False,
        structure_encoder_dim=768,
        freeze_structure_encoder=False,
        train_moe=False,              # 🔥 新增: 训练MoE模式
        structure_encoder_ckpt=None,  # 🔥 新增: 允许载入你在Step 1训练好的权重
    ):
        super().__init__()
        
        model_configs = self.parse_model_configs(model_paths, model_id_with_origin_paths, enable_fp8_training=False)
        
        for config in model_configs:
            config.skip_download = True
        
        if audio_processor_config is not None:
            audio_processor_config = ModelConfig(
                model_id=audio_processor_config.split(":")[0], 
                origin_file_pattern=audio_processor_config.split(":")[1],
                skip_download=True
            )
        
        self.pipe = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16, 
            device="cpu",
            model_configs=model_configs, 
            tokenizer_config=None,
            audio_processor_config=audio_processor_config,
            redirect_common_files=False
        )
        
        if tokenizer_path:
            # print(f"[DEBUG] Initializing tokenizer from: {tokenizer_path}")
            self.pipe.prompter.fetch_tokenizer(tokenizer_path)
            # print(f"[DEBUG] Tokenizer initialized: {hasattr(self.pipe.prompter, 'tokenizer')}")
            if self.pipe.text_encoder is not None:
                self.pipe.prompter.fetch_models(self.pipe.text_encoder)
                # print(f"[DEBUG] Text encoder linked to prompter")
        else:
            print("[WARNING] No tokenizer_path provided! Training may fail.")
        
        self.switch_pipe_to_training_mode(
            self.pipe, trainable_models,
            lora_base_model, lora_target_modules, lora_rank, lora_checkpoint=lora_checkpoint,
            enable_fp8_training=False,
        )
        
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload
        self.extra_inputs = extra_inputs.split(",") if extra_inputs is not None else []
        self.max_timestep_boundary = max_timestep_boundary
        self.min_timestep_boundary = min_timestep_boundary
        self.use_structure_encoder = use_structure_encoder
        self.structure_encoder = None
        
        if use_structure_encoder:
            print(f"[INFO] Initializing Structure Encoder (dim={structure_encoder_dim})...")
            self.structure_encoder = WanAdaptedDualCondEncoder(
                seg_channels=3,
                dit_dim=self.pipe.dit.dim if self.pipe.dit else 5120,
                structure_dim=structure_encoder_dim,
                num_categories=12
            ).to(device="cpu", dtype=torch.bfloat16)

            # 🔥 加载第一阶段训练好的权重
            if structure_encoder_ckpt and os.path.exists(structure_encoder_ckpt):
                from safetensors.torch import load_file
                print(f"[INFO] ✨ Loading trained Structure Encoder from: {structure_encoder_ckpt}")
                state_dict = load_file(structure_encoder_ckpt)
                self.structure_encoder.load_state_dict(state_dict)
            
            # 集成到pipeline
            integrate_structure_encoder_to_pipeline(self.pipe, self.structure_encoder, freeze_dit=False)
            
            # 🔥🔥🔥 根据所处阶段控制梯度 (Phase 2: Train MoE)
            if train_moe:
                print("[INFO] ========================================")
                print("[INFO] TRAINING PHASE 2: MoE Only (Structure Encoder is FROZEN)")
                print("[INFO] ========================================")
                inject_moe_into_dit(self.pipe.dit)
                if self.pipe.dit2 is not None:
                    inject_moe_into_dit(self.pipe.dit2)

                # 1. 冻结所有主模型网络及特征编码器
                for param in self.pipe.parameters(): param.requires_grad = False
                for param in self.structure_encoder.parameters(): param.requires_grad = False

                # 2. 仅解封 MoE Wrapper 里的新参数
                moe_params_count = 0
                for block in self.pipe.dit.blocks:
                    if hasattr(block, 'cross_attn') and type(block.cross_attn).__name__ == 'MoECrossAttentionWrapper':
                        for name, param in block.cross_attn.named_parameters():
                            # 唯独不动 `expert_semantic` （它是原始 WanVideo 知识）
                            if "expert_semantic" not in name:
                                param.requires_grad = True
                                moe_params_count += param.numel()
                
                print(f"[INFO] ✓ MoE Params unfrozen: {moe_params_count/1e6:.2f} M")
                print("[INFO] ========================================")

            # Phase 1: Train Structure Encoder
            elif not freeze_structure_encoder:
                print("[INFO] ========================================")
                print("[INFO] TRAINING PHASE 1: Structure Encoder Only")
                print("[INFO] ========================================")
                # 冻结DiT
                if self.pipe.dit is not None:
                    for param in self.pipe.dit.parameters():
                        param.requires_grad = False
                    print("[INFO] ✓ DiT frozen (14.29B params)")
                
                # 冻结VAE
                if self.pipe.vae is not None:
                    for param in self.pipe.vae.parameters():
                        param.requires_grad = False
                    print("[INFO] ✓ VAE frozen")
                
                # 冻结Text Encoder
                if self.pipe.text_encoder is not None:
                    for param in self.pipe.text_encoder.parameters():
                        param.requires_grad = False
                    print("[INFO] ✓ Text Encoder frozen")
                
                # 冻结其他组件
                if self.pipe.motion_controller is not None:
                    for param in self.pipe.motion_controller.parameters():
                        param.requires_grad = False
                    print("[INFO] ✓ Motion Controller frozen")
                
                if self.pipe.vace is not None:
                    for param in self.pipe.vace.parameters():
                        param.requires_grad = False
                    print("[INFO] ✓ VACE frozen")
                
                # 确保结构编码器可训练
                for param in self.structure_encoder.parameters():
                    param.requires_grad = True
                
                # 统计参数
                total_params = sum(p.numel() for p in self.pipe.parameters())
                trainable_params = sum(p.numel() for p in self.pipe.parameters() if p.requires_grad)
                structure_params = sum(p.numel() for p in self.structure_encoder.parameters())
                
                print(f"[INFO] Total params: {total_params/1e9:.2f}B")
                print(f"[INFO] Trainable params: {trainable_params/1e6:.2f}M (only structure encoder)")
                print(f"[INFO] Structure encoder params: {structure_params/1e6:.2f}M")
                print(f"[INFO] Memory saving: ~{(1 - trainable_params/total_params)*100:.1f}%")
                print("[INFO] ========================================")
            else:
                print("[INFO] Structure encoder frozen (using for inference)")
        else:
            self.structure_encoder = None
        
        
        
    def forward_preprocess(self, data):
        inputs_posi = {"prompt": data["prompt"]}
        inputs_nega = {}
        
        inputs_shared = {
            "input_video": data["video"],
            "height": data["video"][0].size[1],
            "width": data["video"][0].size[0],
            "num_frames": len(data["video"]),
            "cfg_scale": 1,
            "tiled": False,
            "rand_device": self.pipe.device,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "use_gradient_checkpointing_offload": self.use_gradient_checkpointing_offload,
            "cfg_merge": False,
            "vace_scale": 1,
            "max_timestep_boundary": self.max_timestep_boundary,
            "min_timestep_boundary": self.min_timestep_boundary,
        }
        
        # 🆕 处理结构条件（sketch + seg）
        if self.use_structure_encoder:
            if "sketch" in data and "seg" in data:
                inputs_shared["sketch_image"] = data["sketch"]
                inputs_shared["seg_image"] = data["seg"]
            else:
                print("[WARNING] Structure encoder enabled but sketch/seg not in data!")
        
        for extra_input in self.extra_inputs:
            if extra_input in data:
                if extra_input == "input_audio":
                    inputs_shared[extra_input] = data[extra_input]["array"]
                    inputs_shared["audio_sample_rate"] = data[extra_input]["sampling_rate"]
                else:
                    inputs_shared[extra_input] = data[extra_input]
                    
        return {**inputs_shared, **inputs_posi}
    
    def forward(self, data, inputs=None):

        # print(f"[DEBUG] data keys: {list(data.keys())}")

        if inputs is None:
            inputs = self.forward_preprocess(data)
        models = {name: getattr(self.pipe, name) for name in self.pipe.in_iteration_models}
        loss = self.pipe.training_loss(**models, **inputs)
        return loss


def setup_distributed():
    """初始化分布式训练环境 - 全程使用Gloo"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        
        dist.init_process_group(
            backend='gloo',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return rank, world_size, local_rank
    else:
        print('Running in single GPU mode')
        return 0, 1, 0


def cleanup_distributed():
    """清理分布式训练环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def average_gradients(model, world_size):
    """手动实现梯度平均 - 使用Gloo的all_reduce"""
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= world_size


def launch_training_task_torchrun(
    dataset,
    model,
    model_logger,
    args
):
    """使用Gloo+手动梯度同步+8-bit Adam的分布式训练"""
    
    rank, world_size, local_rank = setup_distributed()
    is_main_process = (rank == 0)
    
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
        if is_main_process:
            print(f"Using device: {device}, World size: {world_size}")
            if world_size > 1:
                print(f"Using Gloo backend for all communication (stable for 14B model)")
    else:
        raise RuntimeError("CUDA is not available!")
    
    if model is None:
        if is_main_process:
            print("Loading model on CPU...")
        
        model = WanTrainingModule(
            model_paths=args.model_paths,
            model_id_with_origin_paths=args.model_id_with_origin_paths,
            audio_processor_config=args.audio_processor_config,
            tokenizer_path=args.tokenizer_path,
            trainable_models=args.trainable_models,
            lora_base_model=args.lora_base_model,
            lora_target_modules=args.lora_target_modules,
            lora_rank=args.lora_rank,
            lora_checkpoint=args.lora_checkpoint,
            use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
            extra_inputs=args.extra_inputs,
            max_timestep_boundary=args.max_timestep_boundary,
            min_timestep_boundary=args.min_timestep_boundary,
            # 🆕 添加结构编码器及MoE参数
            use_structure_encoder=args.use_structure_encoder,
            structure_encoder_dim=args.structure_encoder_dim,
            freeze_structure_encoder=args.freeze_structure_encoder,
            train_moe=args.train_moe,
            structure_encoder_ckpt=args.structure_encoder_ckpt,
        )
        
        if is_main_process:
            print("Model loaded on CPU!")
        
        if world_size > 1:
            if is_main_process:
                print("Synchronizing after CPU load...")
            dist.barrier()
            if is_main_process:
                print("CPU synchronization complete!")
        
        if is_main_process:
            print(f"Moving model to {device}...")
        
        model = model.to(device)
        torch.cuda.empty_cache()
        
        if is_main_process:
            print(f"Model moved to {device}!")
        
        if world_size > 1:
            if is_main_process:
                print("Synchronizing after GPU transfer...")
            dist.barrier()
            if is_main_process:
                print("GPU transfer complete!")
                print("Ready for training with manual gradient synchronization (no DDP wrapper)")
    
    # 创建数据加载器
    if world_size > 1:
        sampler = DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=rank, 
            shuffle=True,
            drop_last=True
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1,
            sampler=sampler,
            collate_fn=lambda x: x[0], 
            num_workers=args.dataset_num_workers,
            pin_memory=True,
            persistent_workers=True if args.dataset_num_workers > 0 else False
        )
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=1,
            shuffle=True, 
            collate_fn=lambda x: x[0], 
            num_workers=args.dataset_num_workers,
            pin_memory=True,
            persistent_workers=True if args.dataset_num_workers > 0 else False
        )
    
    if is_main_process:
        print(f"Dataset: {len(dataset)} samples")
        print(f"Dataloader: {len(dataloader)} batches/epoch")
        print(f"Gradient accumulation: {args.gradient_accumulation_steps} steps")
        print(f"Effective batch size: {world_size * args.gradient_accumulation_steps}")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if is_main_process:
        total_params = sum(p.numel() for p in trainable_params)
        print(f"Trainable parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    
    # ⭐⭐⭐ 使用8-bit Adam优化器
    if USE_8BIT_ADAM:
        if is_main_process:
            print("[INFO] Using bitsandbytes 8-bit AdamW")
            print("[INFO] Optimizer memory: ~28GB per device (75% reduction)")
        optimizer = bnb.optim.AdamW8bit(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    else:
        if is_main_process:
            print("[WARNING] Using standard AdamW (may OOM!)")
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
    
    total_steps = args.num_epochs * len(dataloader) // args.gradient_accumulation_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=args.learning_rate * 0.1
    )
    
    if is_main_process:
        print(f"\n{'='*60}")
        print(f"Training Configuration:")
        print(f"  Epochs: {args.num_epochs}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Weight decay: {args.weight_decay}")
        print(f"  Total steps: {total_steps}")
        print(f"  Save every: {args.save_steps} steps")
        print(f"  Output: {args.output_path}")
        print(f"{'='*60}\n")
    
    if world_size > 1:
        if is_main_process:
            print("Final synchronization before training...")
        dist.barrier()
        if is_main_process:
            print("Starting training loop!\n")
    
    # 训练循环
    global_step = 0
    optimizer.zero_grad()
    
    # 🔥🔥🔥 [新增] 初始化 Loss 记录文件 (只在 Rank 0 主进程执行)
    import datetime, csv, os
    loss_log_path = os.path.join(args.output_path, "training_loss_log.csv")
    if torch.distributed.get_rank() == 0:
        os.makedirs(args.output_path, exist_ok=True)
        # 如果是接着训，可以选择追加模式 'a'；如果是新训，可以用 'w'。这里新建一个带表头的
        with open(loss_log_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Time", "Epoch", "Global_Step", "Loss", "LR"])
            
    # 【修复 Bug】：统一使用 epoch 作为变量名
    for epoch in range(args.num_epochs):
        if world_size > 1:
            sampler.set_epoch(epoch)
        
        model.train()
        
        if is_main_process:
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        else:
            pbar = dataloader
        
        epoch_loss = 0.0
        num_items = 0
        
        for step, data in enumerate(pbar):
            try:
                loss = model.forward(data)
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
                
                # 记录真实的未放缩 loss
                real_loss = loss.item() * args.gradient_accumulation_steps
                epoch_loss += real_loss
                num_items += 1
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if world_size > 1:
                        average_gradients(model, world_size)
                    
                    if hasattr(args, 'max_grad_norm') and args.max_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    current_lr = scheduler.get_last_lr()[0]
                    
                    if is_main_process and hasattr(pbar, 'set_postfix'):
                        pbar.set_postfix({
                            'loss': f'{epoch_loss/num_items:.4f}',
                            'lr': f'{current_lr:.2e}',
                            'step': global_step
                        })
                    
                    # 🔥🔥🔥 [新增] 将当前的 Loss 保存到 CSV 中 (按全局有效步数保存)
                    if is_main_process:
                        with open(loss_log_path, mode='a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            
                            # 获取与终端完全一致的累计平均 Loss
                            running_avg_loss = epoch_loss / num_items 
                            
                            # 写入平滑后的 running_avg_loss
                            writer.writerow([current_time, epoch + 1, global_step, round(running_avg_loss, 6), f"{current_lr:.2e}"])
                    
                    if args.save_steps and global_step % args.save_steps == 0 and is_main_process:
                        save_checkpoint(model, model_logger, global_step, args)
                        
            except Exception as e:
                print(f"[Rank {rank}] Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if is_main_process:
            avg_loss = epoch_loss / num_items if num_items > 0 else 0
            print(f"\nEpoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
            
            if not args.save_steps:
                save_checkpoint(model, model_logger, epoch, args, epoch=True)
    
    if is_main_process:
        print("\nTraining completed!")
        save_checkpoint(model, model_logger, global_step, args, final=True)
    
    cleanup_distributed()


def save_checkpoint(model, model_logger, step_or_epoch, args, epoch=False, final=False):
    """保存检查点"""
    try:
        state_dict = model.state_dict()
        state_dict = model.export_trainable_state_dict(
            state_dict, 
            remove_prefix=model_logger.remove_prefix_in_ckpt
        )
        
        os.makedirs(model_logger.output_path, exist_ok=True)
        
        if final:
            filename = f"final_step-{step_or_epoch}.safetensors"
        elif epoch:
            filename = f"epoch-{step_or_epoch+1}.safetensors"
        else:
            filename = f"step-{step_or_epoch}.safetensors"
        
        save_path = os.path.join(model_logger.output_path, filename)
        
        from safetensors.torch import save_file
        save_file(state_dict, save_path)
        
        print(f"✓ Checkpoint saved: {filename}")
        
        latest_path = os.path.join(model_logger.output_path, "latest.safetensors")
        if os.path.exists(latest_path):
            os.remove(latest_path)
        import shutil
        shutil.copy2(save_path, latest_path)
        
    except Exception as e:
        print(f"✗ Error saving checkpoint: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = wan_parser()
    
    existing_args = {opt for action in parser._actions for opt in action.option_strings}
    
    # 🆕 添加结构编码器相关参数
    if '--use_structure_encoder' not in existing_args:
        parser.add_argument("--use_structure_encoder", action="store_true", 
                          help="Enable dual-stream structure encoder (sketch+seg)")
    if '--structure_encoder_dim' not in existing_args:
        parser.add_argument("--structure_encoder_dim", type=int, default=768,
                          help="Structure encoder output dimension")
    if '--freeze_structure_encoder' not in existing_args:
        parser.add_argument("--freeze_structure_encoder", action="store_true",
                          help="Freeze structure encoder weights")
    
    # 🔥 新增参数
    if '--train_moe' not in existing_args:
        parser.add_argument("--train_moe", action="store_true", help="Enable MoE mode and freeze base structures")
    if '--structure_encoder_ckpt' not in existing_args:
        parser.add_argument("--structure_encoder_ckpt", type=str, default=None, help="Path to Step 1 encoder checkpoint")
    
    if '--max_grad_norm' not in existing_args:
        parser.add_argument("--max_grad_norm", type=float, default=None, 
                          help="Max gradient norm for clipping")
    if '--tokenizer_path' not in existing_args:
        parser.add_argument("--tokenizer_path", type=str, default=None, 
                          help="Path to local tokenizer directory")
    
    args = parser.parse_args()
    
    if args.tokenizer_path is None:
        if args.model_id_with_origin_paths:
            first_model = args.model_id_with_origin_paths.split(",")[0]
            base_dir = first_model.split(":")[0]
            args.tokenizer_path = os.path.join(base_dir, "google/umt5-xxl")
            print(f"[INFO] Auto-detected tokenizer path: {args.tokenizer_path}")
            
            if not os.path.exists(args.tokenizer_path):
                print(f"[WARNING] Tokenizer path does not exist: {args.tokenizer_path}")
                print(f"[WARNING] Please manually specify --tokenizer_path")
        else:
            print("[WARNING] Cannot auto-detect tokenizer path without model_id_with_origin_paths")
    
    print(f"\n{'='*60}")
    print(f"Wan Video Training with PyTorch DDP (torchrun)")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_base_path}")
    print(f"Output: {args.output_path}")
    print(f"Tokenizer: {args.tokenizer_path}")
    print(f"{'='*60}\n")
    
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),  # 应该是 "video,prompt,sketch,seg"
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            # 🆕 添加sketch和seg的加载器
            "sketch": ToAbsolutePath(args.dataset_base_path) >> LoadImage(),
            "seg": ToAbsolutePath(args.dataset_base_path) >> LoadImage(),
        }
    )
    
    model = None
    
    model_logger = ModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt
    )
    
    launch_training_task_torchrun(dataset, model, model_logger, args=args)