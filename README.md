# video_gen (Wan TI2V Standalone)

本目录是可独立运行的 Wan2.2 TI2V 仓库，不再依赖 DiffSynth-change。

## 关键内容
- diffsynth/
  - 已裁剪为 Wan-only 最小闭包（仅保留 TI2V 训练/推理必要模块）
- examples/wanvideo/
  - 已完整纳入 Wan 训练与验证脚本
- examples/wanvideo/dual_cond_encoder.py
  - 创新点1：双流结构编码器（Sketch + Seg）
- examples/wanvideo/moe_attention.py
  - 创新点2：四专家 MoE Cross-Attention（含路由与时空干预）
- diffsynth/pipelines/wan_video_new.py
  - 创新点3：动态时空掩码控制链路接入

## 环境安装
1. 安装基础依赖
   - pip install -r requirements.txt
2. 安装可选加速依赖（按需）
   - pip install -r requirements-optional.txt
3. 一键安装并执行自检
  - bash scripts/bootstrap_env.sh
  - INSTALL_OPTIONAL=1 bash scripts/bootstrap_env.sh

## 启动前自检
- 执行：python scripts/self_check.py
- 自检项：
  - 关键文件存在性
  - 核心 Python 文件语法可编译
  - 必需外部依赖导入
  - 仓库内部导入链
  - 可选数据路径校验（--dataset-base-path, --dataset-metadata-path）

## 已做的瘦身检查
- 已移除与 Wan 主链路无关的目录：controlnets、processors、extensions、tokenizer_configs 等
- 已移除非 Wan 的模型实现文件（SD/SDXL/Flux/Hunyuan/Qwen 等）
- `ModelManager` 改为 Wan-only 检测与加载，不再依赖全量模型注册
- 保留的 `diffsynth` 文件数从 534 降到 32

## 训练启动
1. 推荐入口（先自检，再 torchrun）
   - bash scripts/run_train_ti2v5b.sh -- <训练参数>
2. 仅自检，不启动训练
   - SELF_CHECK_ONLY=1 bash scripts/run_train_ti2v5b.sh

## 训练建议
1. Phase 1：只训练结构编码器
2. Phase 2：冻结结构编码器，训练 MoE
3. 使用 --resume_from_checkpoint 断点续训

## 备注
仓库已可独立运行。建议每次训练前先运行 scripts/self_check.py，防止导入链回归问题。