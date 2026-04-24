import torch
import torch.nn as nn
import torch.nn.functional as F

class MoECrossAttentionWrapper(nn.Module):
    """
    四元混合专家交叉注意力 (无损包裹版)
    利用原有预训练 CrossAttention 作为 Expert 1，其余 Expert 零初始化。
    """
    def __init__(self, orig_cross_attn: nn.Module, dim: int, num_heads: int):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        
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

        # --- MoE Router (路由器) ---
        self.router = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.LayerNorm(dim // 4),
            nn.SiLU(),
            nn.Linear(dim // 4, 4) 
        )
        
        # 🔥 关键：设置高 bias 给 Expert 1，使其初始输出概率接近 1.0
        nn.init.zeros_(self.router[-1].weight)
        nn.init.constant_(self.router[-1].bias, 0.0)
        self.router[-1].bias.data[0] = 5.0 # softmax([5,0,0,0]) ≈ [0.99, 0.0, 0.0, 0.0]

        # 🔥 新增代码：在 init 的最底部，强制将新初始化的 MoE 模块转换为主干模型对应的 dtype (BFloat16) 和 所在 Device
        target_device = next(orig_cross_attn.parameters()).device
        target_dtype = next(orig_cross_attn.parameters()).dtype
        self.to(device=target_device, dtype=target_dtype)

    def forward(self, x, context, sketch_context=None, **kwargs):
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

        # 5. 路由打分与加权
        router_logits = self.router(x) # [B, L, 4]
        weights = F.softmax(router_logits, dim=-1)
        
        w_sem = weights[..., 0:1]
        w_struct = weights[..., 1:2]
        w_fusion = weights[..., 2:3]
        w_self = weights[..., 3:4]

        # 🔥 放宽探针：使用 print 并将随机概率提高到 5% (或者您只训练了几步的话，可以直接改为 1.0 也就是全打印)
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

def inject_moe_into_dit(dit_model: nn.Module):
    """
    网络手术：遍历所有的 DiT Block，把交叉注意力替换为包裹好的 MoE 模块
    """
    import logging
    
    count = 0
    for block in dit_model.blocks:
        if hasattr(block, 'cross_attn'):
            orig_cross_attn = block.cross_attn
            
            # 从原始注意力机制中获取 dim 和 num_heads
            dim = orig_cross_attn.dim
            num_heads = orig_cross_attn.num_heads
            
            # 替换为新模块
            block.cross_attn = MoECrossAttentionWrapper(orig_cross_attn, dim, num_heads)
            count += 1
            
    logging.info(f"[MoE Injection] Successfully injected MoE into {count} DiT blocks.")