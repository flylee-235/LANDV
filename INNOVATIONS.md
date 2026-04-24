# Innovations (Paper-Oriented Summary: 强化版)

本项目在现有视频扩散模型（DiT）的基础上，针对风格化（尤其是水墨风格）、结构可控性与时空动态解耦设计了三大核心创新。所有创新均通过多阶段训练协议进行无损挂载。

## Innovation 1: 双流拓扑-语义编码器 (Dual-Stream Topological-Semantic Encoder)
**核心思想**：传统的单一流条件编码往往丢失高频线条或低频语义。本模块引入独立的两条特征流：一条提取拓扑连通性（线稿 Sketch），另一条提取区域语义分布（分割图 Seg），在特征层进行跨流注意力交互，最终与文本提示（Text Prompt）拼接，作为 DiT 的拓展上下文。

**技术实现细节**：
1. **显式类别语义查表**：`SemanticCategoryEncoder` 将输入的分割 RGB 图通过 L2 距离映射到预定义的 12 种自然语义调色板（水、山、云、人等），生成概率分布 `probs` 与全局/局部语义特征。
2. **跨流交互 (Cross-Stream Interaction)**：通过线稿与分割特征的交叉多头注意力（`sketch_to_seg_attn` 与 `seg_to_sketch_attn`）实现高低频特征互补。
3. **架构**：Sketch 使用 ResNet18 + Swin 混合底座，Seg 使用 Swin 底座。

**代码定位**：
*   **文件**: dual_cond_encoder.py
*   **关键类**: `SemanticCategoryEncoder`, `CrossStreamInteraction`, `WanAdaptedDualCondEncoder`

---

## Innovation 2: 运动学引导的非对称 MoE 交叉注意力 (Kinematic-Guided Asymmetric MoE)
**核心思想**：为了在不破坏预训练模型（如 WanVideo）强大生成先验的前提下注入严密的结构控制，将原生 DiT Block 的 `CrossAttention` 升级为 4 专家（Semantic, Structure, Fusion, Self）混合路由模块。

**技术实现细节**：
1. **四专家分工**：
    *   *Semantic Expert*：保留主干模型原预训练权重的交叉注意力。
    *   *Structure Expert*：接受 Innovation 1 提取的 `sketch_context`，零初始化以保证早期训练稳定。
    *   *Fusion / Self Expert*：以 MLP 形式分别处理联合特征和自特征。
2. **双路由与水墨先验调制**：
    *   **Token Router**: 逐 Token进行路由打分。
    *   **Global Style Router**: 基于全局图像和语义上下文输出风格调制分数。
    *   **Ink Prior**: 根据笔触强度和动态衰减强度动态偏移各专家概率（笔触强提高结构约束，动态强提高融合性）。
3. **非对称初始化与稀疏推理**：偏置偏向 Semantic 专家，推理时执行 Top-2 稀疏掩码，使专家分工高度明晰。

**代码定位**：
*   **文件**: moe_attention.py
*   **关键机制**: `MoECrossAttentionWrapper`, `_compute_ink_prior`, `inject_moe_into_dit`（网络手术挂载逻辑）

---

## Innovation 3: 解耦时空调控场 (DSRF) 与路由感知正则 (RASR) 
**核心思想**：解决长视频生成中“结构僵化”问题（例如水墨画中“山需要保持静态线稿，而水和云应随时间流动淡出线稿约束”的“山静水动”需求）。将空间分类区域（Spatial Gate）与时间衰减曲线（Temporal Curve）解耦计算，在路由层进行自适应降权。

**技术实现细节**：
1. **上游生成 (类别感知动态曲线)**：`EnhancedDSRFHead` 基于全局语义生成基础时间约束曲线，利用 `dynamic_mask`（提取动态类别：水、云、船等）过滤不可动的静物，通过一维卷积输出平滑递减的时序特征 `time_decay_curve`，并附带平滑与单调递减正则约束（`dsrf_aux_loss`）。
2. **中游传递 (时空场融合)**：Pipeline 内部将 `dynamic_mask` 和 `time_decay_curve` 进行外积扩张，构建出形状为 `[B, F, H, W, 1]` 的三维张量 `st_decay`。
3. **下游重分配与 RASR 正则**：在 MoE 注意力中，检测到 `st_decay` 时，按比率降低结构专家 (`w_struct`) 权重，并将溢出权重均分给语义和融合专家。此外，计算 `rasr_loss`：惩罚路由塌缩（信息熵）与无视时间衰减的异常结构高权重。

**代码定位**：
*   **上游生成**: `examples/wanvideo/dual_cond_encoder.py` -> `EnhancedDSRFHead.forward` (计算 `time_decay_curve`与正则)
*   **中游传递**: `diffsynth/pipelines/wan_video_new.py` -> `model_fn_wan_video` (构建 `st_decay` 时空解耦场)
*   **终点分配**: `examples/wanvideo/moe_attention.py` -> `MoECrossAttentionWrapper.forward` (削减 `w_struct` 并累加 `rasr_loss`)
*   **损失合并**: `examples/wanvideo/model_training/train_torchrun.py` -> `train_moe` 分支 (`loss = loss + 0.02 * aux_loss + 0.01 * dsrf_aux`)