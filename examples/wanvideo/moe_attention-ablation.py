import torch
import torch.nn as nn
import torch.nn.functional as F

class MoECrossAttentionWrapper(nn.Module):
    """
    四元混合专家交叉注意力 (无损包裹版)
    利用原有预训练 CrossAttention 作为 Expert 1，其余 Expert 零初始化。
    """
    def __init__(
        self,
        orig_cross_attn: nn.Module,
        dim: int,
        num_heads: int,
        enable_style_router: bool = True,
        enable_ink_prior: bool = True,
        enable_dsrf: bool = True,
        enable_top2_infer: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # ablation 开关
        self.enable_style_router = enable_style_router
        self.enable_ink_prior = enable_ink_prior
        self.enable_dsrf = enable_dsrf
        self.enable_top2_infer = enable_top2_infer

        # --- Expert 1: 原版语义专家 (保留全部预训练权重) ---
        self.expert_semantic = orig_cross_attn

        # --- Expert 2: 结构专家 (Structure) 处理 sketch_context ---
        self.k_struct = nn.Linear(dim, dim)
        self.v_struct = nn.Linear(dim, dim)
        self.o_struct = nn.Linear(dim, dim) # 🔥 新增结构输出投影
        self.norm_k_struct = nn.LayerNorm(dim)

        # 初始化为0，确保初期不扰乱网络
        nn.init.zeros_(self.k_struct.weight)
        nn.init.zeros_(self.v_struct.weight)
        nn.init.zeros_(self.o_struct.weight)
        nn.init.zeros_(self.o_struct.bias)

        # --- Expert 3: 融合专家 (Fusion MLP) ---
        self.expert_fusion = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )
        self.expert_fusion[-1].weight.data.zero_() # Zero-init

        # --- Expert 4: 自我修正专家 (Self MLP) ---
        self.expert_self = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.expert_self[-1].weight.data.zero_() # Zero-init

        # --- 原 Token Router ---
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 4)
        )

        # --- 新增：全局风格 Router（基于语义上下文+结构上下文的全局统计）---
        self.style_router = nn.Sequential(
            nn.Linear(dim * 2, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 4)
        )

        # 融合强度/温度/先验强度（可学习）
        self.style_blend_alpha = nn.Parameter(torch.tensor(0.0))   # sigmoid 后约 0.5
        self.router_temperature = nn.Parameter(torch.tensor(1.0))  # softmax 温度
        self.prior_log_scale = nn.Parameter(torch.tensor(0.8))     # 水墨先验强度

        # 初始偏向 semantic expert
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 0.0)
        self.router[-1].bias.data[0] = 5.0

        # style_router 初始弱作用，防止破坏预训练先验
        nn.init.zeros_(self.style_router[-1].weight)
        nn.init.constant_(self.style_router[-1].bias, 0.0)
        self.style_router[-1].bias.data[0] = 0.5
        self.style_router[-1].bias.data[2] = 0.3

        # 🔥 新增代码：在 init 的最底部，强制将新初始化的 MoE 模块转换为主干模型对应的 dtype (BFloat16) 和 所在 Device
        target_device = next(orig_cross_attn.parameters()).device
        target_dtype = next(orig_cross_attn.parameters()).dtype
        self.to(device=target_device, dtype=target_dtype)

    def _compute_ink_prior(self, sketch_context, st_decay, eps=1e-6):
        """
        水墨任务先验：
        - 笔触强 -> 结构专家权重上升
        - 动态强(st_decay高) -> 融合/语义上升，结构下降
        """
        stroke_strength = torch.tanh(sketch_context.float().abs().mean(dim=(1, 2)) * 2.0)  # [B]
        if st_decay is not None:
            motion_strength = st_decay.float().mean(dim=(1, 2)).clamp(0, 1)  # [B]
        else:
            motion_strength = torch.zeros_like(stroke_strength)

        p_sem = (1.0 - motion_strength) + 0.15
        p_struct = stroke_strength * (1.0 - motion_strength) + 0.05
        p_fusion = motion_strength + 0.3 * stroke_strength + 0.05
        p_self = torch.full_like(p_sem, 0.08)

        prior = torch.stack([p_sem, p_struct, p_fusion, p_self], dim=-1)  # [B,4]
        prior = prior / (prior.sum(dim=-1, keepdim=True) + eps)
        return prior

    # 🔥 forward 接收 st_decay 参数
    def forward(self, x, context, sketch_context=None, st_decay=None, **kwargs):
        # 1. 运行原版的语义专家
        out_sem = self.expert_semantic(x, context, **kwargs)

        # 2. 如果没有提供结构条件，直接返回原版结果
        if sketch_context is None:
            return out_sem

        # 3. 计算 Expert 2 (包含精度对齐和输出投影)
        B, L_q, D = x.shape
        q = x.view(B, L_q, self.num_heads, D // self.num_heads).transpose(1, 2)

        k_s = self.norm_k_struct(self.k_struct(sketch_context))
        v_s = self.v_struct(sketch_context)
        k_s = k_s.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)
        v_s = v_s.view(B, -1, self.num_heads, D // self.num_heads).transpose(1, 2)

        # 对齐精度以防 Float32/BF16 冲突
        out_struct = F.scaled_dot_product_attention(
            q.to(v_s.dtype), k_s.to(v_s.dtype), v_s
        )
        out_struct = out_struct.transpose(1, 2).reshape(B, L_q, D)
        out_struct = self.o_struct(out_struct) # 投影映射

        # 4. 计算 Expert 3 & 4
        out_fusion = self.expert_fusion(out_sem + out_struct)
        out_self = self.expert_self(x)

        # 5. 路由打分与机制干预 (Dynamic Kinematic Intervention)
        token_logits = self.router(x)  # [B, L, 4]

        router_logits = token_logits

        if self.enable_style_router:
            global_ctx = torch.cat([context.mean(dim=1), sketch_context.mean(dim=1)], dim=-1)  # [B, 2D]
            style_logits = self.style_router(global_ctx).unsqueeze(1)  # [B,1,4]
            alpha = torch.sigmoid(self.style_blend_alpha)
            router_logits = router_logits + alpha * style_logits

        if self.enable_ink_prior:
            ink_prior = self._compute_ink_prior(sketch_context, st_decay).unsqueeze(1)  # [B,1,4]
            router_logits = router_logits + self.prior_log_scale * torch.log(ink_prior + 1e-6).to(router_logits.dtype)

        temp = torch.clamp(self.router_temperature, 0.6, 1.8)
        weights = F.softmax(router_logits / temp, dim=-1)

        # 推理时 Top-2 稀疏化
        if (not self.training) and self.enable_top2_infer:
            topv, topi = weights.topk(2, dim=-1)
            sparse_mask = torch.zeros_like(weights).scatter_(-1, topi, 1.0)
            weights = weights * sparse_mask
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        w_sem = weights[..., 0:1]
        w_struct = weights[..., 1:2]
        w_fusion = weights[..., 2:3]
        w_self = weights[..., 3:4]

        # DSRF 重分配
        if self.enable_dsrf and (st_decay is not None):
            w_struct_reduced = w_struct * (1.0 - st_decay)
            weight_diff = w_struct - w_struct_reduced
            w_sem = w_sem + weight_diff * 0.5
            w_fusion = w_fusion + weight_diff * 0.5
            w_struct = w_struct_reduced

        # 放宽探针：使用 print 并将随机概率提高到 5% (或者您只训练了几步的话，可以直接改为 1.0 也就是全打印)
        if self.training and torch.rand(1).item() < 0.001:
            print(f"\n[MoE Probe] Router Avg Weights | Sem(Base): {w_sem.mean().item():.3f} "
                  f"| Struct(Sketch): {w_struct.mean().item():.3f} "
                  f"| Fusion: {w_fusion.mean().item():.3f} "
                  f"| Self: {w_self.mean().item():.3f}")

        fused_out = (out_sem * w_sem +
                     out_struct * w_struct +
                     out_fusion * w_fusion +
                     out_self * w_self)

        return fused_out

def inject_moe_into_dit(
    dit_model: nn.Module,
    enable_style_router: bool = True,
    enable_ink_prior: bool = True,
    enable_dsrf: bool = True,
    enable_top2_infer: bool = True,
):
    """
    网络手术：遍历所有的 DiT Block，把交叉注意力替换为包裹好的 MoE 模块
    """
    import logging

    count = 0
    for block in dit_model.blocks:
        if hasattr(block, 'cross_attn'):
            orig_cross_attn = block.cross_attn
            dim = orig_cross_attn.dim
            num_heads = orig_cross_attn.num_heads

            block.cross_attn = MoECrossAttentionWrapper(
                orig_cross_attn=orig_cross_attn,
                dim=dim,
                num_heads=num_heads,
                enable_style_router=enable_style_router,
                enable_ink_prior=enable_ink_prior,
                enable_dsrf=enable_dsrf,
                enable_top2_infer=enable_top2_infer,
            )
            count += 1

    logging.info(f"[MoE Injection] Successfully injected MoE into {count} DiT blocks.")