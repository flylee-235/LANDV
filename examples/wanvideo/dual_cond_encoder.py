import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from einops import rearrange
from typing import Optional, List, Tuple

# 🔥 新增：类感知与空间门控关联的 DSRF 头
class EnhancedDSRFHead(nn.Module):
    def __init__(self, semantic_dim=128, num_categories=12, max_frames=81, dynamic_classes=(2, 3, 7, 8, 10, 11)):
        super().__init__()
        self.max_frames = max_frames
        self.dynamic_classes = list(dynamic_classes)
        self.base_mlp = nn.Sequential(nn.Linear(semantic_dim, 128), nn.GELU(), nn.Linear(128, max_frames))
        self.class_mlp = nn.Sequential(nn.Linear(num_categories, 64), nn.GELU(), nn.Linear(64, len(self.dynamic_classes) * max_frames))
        self.spatial_gate = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 1))
        self.smooth_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        with torch.no_grad():
            self.smooth_conv.weight.fill_(1.0 / 5.0)

    def forward(self, category_global, category_presence, dynamic_mask):
        dtype = category_global.dtype
        B = category_global.shape[0]
        base = self.base_mlp(category_global.float())
        cls = self.class_mlp(category_presence.float()).view(B, len(self.dynamic_classes), self.max_frames)
        dyn_presence = category_presence[:, self.dynamic_classes].float()
        dyn_presence = dyn_presence / (dyn_presence.sum(dim=-1, keepdim=True) + 1e-6)
        dyn_curve = (cls * dyn_presence.unsqueeze(-1)).sum(dim=1)
        gate = torch.sigmoid(self.spatial_gate(dynamic_mask.float())).clamp(0.05, 0.95)
        curve = torch.sigmoid(base + gate * dyn_curve)
        curve = self.smooth_conv(curve.unsqueeze(1)).squeeze(1).clamp(0.0, 1.0)
        
        diff = curve[:, 1:] - curve[:, :-1]
        reg_loss = 0.5 * diff.abs().mean() + 1.0 * F.relu(diff).mean() # 平滑 + 单调递减约束
        return curve.to(dtype), reg_loss.to(dtype)

class CrossStreamInteraction(nn.Module):
    """双流交互模块：让sketch和seg特征相互增强"""
    def __init__(self, sketch_dim: int, seg_dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = max(sketch_dim, seg_dim)
        self.sketch_proj = nn.Linear(sketch_dim, self.dim) if sketch_dim != self.dim else nn.Identity()
        self.seg_proj = nn.Linear(seg_dim, self.dim) if seg_dim != self.dim else nn.Identity()
        
        self.sketch_to_seg_attn = nn.MultiheadAttention(self.dim, num_heads, batch_first=True)
        self.seg_to_sketch_attn = nn.MultiheadAttention(self.dim, num_heads, batch_first=True)
        
        self.sketch_gate = nn.Sequential(nn.Linear(self.dim * 2, self.dim), nn.Sigmoid())
        self.seg_gate = nn.Sequential(nn.Linear(self.dim * 2, self.dim), nn.Sigmoid())
        
        self.sketch_out = nn.Linear(self.dim, sketch_dim)
        self.seg_out = nn.Linear(self.dim, seg_dim)
        
    def forward(self, sketch_feat: torch.Tensor, seg_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        orig_shape_sk = sketch_feat.shape
        orig_shape_seg = seg_feat.shape
        
        # 修复：标准的 PyTorch 形状是 [B, C, H, W]
        if len(sketch_feat.shape) == 4:
            sketch_feat = rearrange(sketch_feat, 'b c h w -> b (h w) c')
            seg_feat = rearrange(seg_feat, 'b c h w -> b (h w) c')
        
        sk_proj = self.sketch_proj(sketch_feat)
        seg_proj = self.seg_proj(seg_feat)
        
        sk_from_seg, _ = self.seg_to_sketch_attn(query=sk_proj, key=seg_proj, value=seg_proj)
        seg_from_sk, _ = self.sketch_to_seg_attn(query=seg_proj, key=sk_proj, value=sk_proj)
        
        sk_gate = self.sketch_gate(torch.cat([sk_proj, sk_from_seg], dim=-1))
        seg_gate = self.seg_gate(torch.cat([seg_proj, seg_from_sk], dim=-1))
        
        sk_enhanced = sk_proj + sk_gate * sk_from_seg
        seg_enhanced = seg_proj + seg_gate * seg_from_sk
        
        sk_enhanced = self.sketch_out(sk_enhanced)
        seg_enhanced = self.seg_out(seg_enhanced)
        
        # 恢复回 [B, C, H, W]
        if len(orig_shape_sk) == 4:
            H, W = orig_shape_sk[2], orig_shape_sk[3]
            sk_enhanced = rearrange(sk_enhanced, 'b (h w) c -> b c h w', h=H, w=W)
            seg_enhanced = rearrange(seg_enhanced, 'b (h w) c -> b c h w', h=H, w=W)
        
        return sk_enhanced, seg_enhanced

class SemanticCategoryEncoder(nn.Module):
    """基于纯颜色Mask映射的语义类别编码器 (12类)"""
    def __init__(
        self,
        num_categories: int = 12,
        category_dim: int = 128,
        palette_rgb: Optional[List[Tuple[int, int, int]]] = None,
        temperature: float = 0.02,
        unknown_threshold: float = 0.20,
    ):
        super().__init__()
        self.num_categories = num_categories
        self.category_dim = category_dim
        self.temperature = temperature
        self.unknown_threshold = unknown_threshold

        if palette_rgb is None:
            palette_rgb = [
                (34, 139, 34),    # Tree
                (139, 69, 19),    # Mountain
                (0, 191, 255),    # Water
                (255, 140, 0),    # Boat
                (178, 34, 34),    # Building
                (210, 105, 30),   # Bridge
                (128, 128, 128),  # Rock
                (255, 0, 255),    # Person
                (240, 248, 255),  # Cloud
                (255, 192, 203),  # Flower
                (255, 215, 0),    # Animal
                (0, 0, 0),        # Unknown / Background (黑)
            ]
        assert len(palette_rgb) == num_categories

        palette_tensor = torch.tensor(palette_rgb, dtype=torch.float32) / 255.0
        self.register_buffer("palette", palette_tensor)
        self.category_embeddings = nn.Parameter(torch.randn(num_categories, category_dim))

    def forward(self, seg_rgb, target_hw=None, hard_assign=True):
        if target_hw is not None:
            seg_rgb = F.interpolate(seg_rgb, size=target_hw, mode='nearest')

        B, C, H, W = seg_rgb.shape
        K = self.num_categories
        
        seg_flat = seg_rgb.permute(0, 2, 3, 1).reshape(-1, 3) # [B*H*W, 3]
        dist2 = torch.cdist(seg_flat, self.palette.to(seg_rgb.dtype), p=2)      # 最好也让palette对齐特征精度
        
        if hard_assign:
            idx = dist2.argmin(dim=-1)
            probs = F.one_hot(idx, num_classes=K).to(self.category_embeddings.dtype)  # <--- 动态获取精度格式
            min_dist = dist2.min(dim=-1, keepdim=True).values
            valid = (min_dist < self.unknown_threshold).to(self.category_embeddings.dtype) # <--- 动态获取精度格式
            probs = probs * valid
        else:
            probs = torch.softmax(-dist2 / max(self.temperature, 1e-6), dim=-1).to(self.category_embeddings.dtype)
            min_dist = dist2.min(dim=-1, keepdim=True).values
            valid = (min_dist < self.unknown_threshold).to(self.category_embeddings.dtype) # <--- 动态获取精度格式
            probs = probs * valid
            
        probs = probs.reshape(B, H, W, K)
        
        spatial_semantic = torch.matmul(probs, self.category_embeddings) # [B, H, W, dim]
        category_presence = probs.reshape(B, -1, K).mean(dim=1)          # [B, K]
        global_semantic = (self.category_embeddings.unsqueeze(0) * category_presence.unsqueeze(-1)).sum(dim=1)

        return {
            "probs": probs,                          # 🔥 新增输出：每个像素类别的概率图 [B, H, W, K]
            "spatial_semantic": spatial_semantic,    # [B, H, W, dim]
            "global_semantic": global_semantic,      # [B, dim]
        }

class WanAdaptedDualCondEncoder(nn.Module):
    """完全适配的新版双流编码器 (删除了冗长融合类，全面接入颜色语义映射)"""
    def __init__(
        self, 
        seg_channels: int = 3,
        dit_dim: int = 5120,
        structure_dim: int = 768,
        num_categories: int = 12,    # <- 已为你改为12类
        output_seq_length: Optional[int] = None
    ):
        super().__init__()
        self.dit_dim = dit_dim
        self.structure_dim = structure_dim
        self.output_seq_length = output_seq_length
        
        # Sketch 编码器 (1通道)
        res_sk = models.resnet18(weights='DEFAULT')
        self.sk_swin_1ch_proj = nn.Sequential(nn.Conv2d(1, 3, kernel_size=1, bias=False), nn.BatchNorm2d(3))
        with torch.no_grad():
            self.sk_swin_1ch_proj[0].weight.data.fill_(1.0)

        swin_sk = models.swin_t(weights='DEFAULT')
        self.sk_cnn_stem = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            self.sk_cnn_stem.weight.data = res_sk.conv1.weight.data.mean(1, keepdim=True)
            
        self.sk_cnn_stem_rest = nn.Sequential(res_sk.bn1, res_sk.relu, res_sk.maxpool)
        self.sk_cnn_l1 = res_sk.layer1  # H/4, 64c
        self.sk_cnn_l2 = res_sk.layer2  # H/8, 128c
        
        self.sk_swin_l1 = nn.Sequential(swin_sk.features[0], swin_sk.features[1])  # H/4, 96c
        self.sk_swin_merge = swin_sk.features[2]
        self.sk_swin_l2 = swin_sk.features[3]  # H/8, 192c
        
        # Seg 编码器
        swin_seg = models.swin_t(weights=None)
        self.seg_embed = nn.Conv2d(seg_channels, 3, kernel_size=1)
        self.seg_swin_l1 = nn.Sequential(swin_seg.features[0], swin_seg.features[1])  # H/4, 96c
        self.seg_swin_merge = swin_seg.features[2]
        self.seg_swin_l2 = swin_seg.features[3]  # H/8, 192c
        
        # 双流交互
        self.cross_interaction_h4 = CrossStreamInteraction(sketch_dim=96, seg_dim=96)
        self.cross_interaction_h8 = CrossStreamInteraction(sketch_dim=192, seg_dim=192)
        
        # 语义类别编码器
        self.category_encoder = SemanticCategoryEncoder(num_categories=num_categories, category_dim=128)
        
        # 直接内置的层级融合逻辑，告别 MultiscaleStructureFusion
        self.fusion_h4 = nn.Sequential(
            nn.Linear(64 + 96 + 96, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, structure_dim)
        )
        self.fusion_h8 = nn.Sequential(
            nn.Linear(128 + 192 + 192, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, structure_dim)
        )
        
        self.to_dit_context = nn.Sequential(
            nn.Linear(structure_dim + 128, structure_dim),
            nn.LayerNorm(structure_dim),
            nn.GELU(),
            nn.Linear(structure_dim, dit_dim)
        )
        
        nn.init.zeros_(self.to_dit_context[-1].weight)
        nn.init.zeros_(self.to_dit_context[-1].bias)
        
        # 🔥 新增：DSRF 动态气韵流转层 (Dynamic Spirit-Resonance Flow)
        # 用全局语义直接推理出时间维度 [81帧] 的衰减渐变曲线
        self.dsrf_layer = EnhancedDSRFHead(semantic_dim=128, num_categories=num_categories, max_frames=81)
        self.dsrf_aux_loss = None

    def forward(self, sketch: torch.Tensor, seg: torch.Tensor, return_pyramid: bool = False) -> torch.Tensor:
        if sketch.shape[1] == 3:
            sketch = 0.299 * sketch[:, 0:1] + 0.587 * sketch[:, 1:2] + 0.114 * sketch[:, 2:3]

        c = self.sk_cnn_stem(sketch)
        c = self.sk_cnn_stem_rest(c)
        c1 = self.sk_cnn_l1(c)      # [B, 64, H/4, W/4]
        c2 = self.sk_cnn_l2(c1)     # [B, 128, H/8, W/8]
        
        sk_3ch = self.sk_swin_1ch_proj(sketch)
        s_sk1 = self.sk_swin_l1(sk_3ch)
        s_sk1 = rearrange(s_sk1, 'b h w c -> b c h w') # 统一为BCHW喂给交叉注意力
        s_sk2 = self.sk_swin_merge(rearrange(s_sk1, 'b c h w -> b h w c'))
        s_sk2 = self.sk_swin_l2(s_sk2)
        s_sk2 = rearrange(s_sk2, 'b h w c -> b c h w')
        
        seg_img = self.seg_embed(seg)
        s_seg1 = self.seg_swin_l1(seg_img)
        s_seg1 = rearrange(s_seg1, 'b h w c -> b c h w')
        s_seg2 = self.seg_swin_merge(rearrange(s_seg1, 'b c h w -> b h w c'))
        s_seg2 = self.seg_swin_l2(s_seg2)
        s_seg2 = rearrange(s_seg2, 'b h w c -> b c h w')
        
        s_sk1_enhanced, s_seg1_enhanced = self.cross_interaction_h4(s_sk1, s_seg1)
        s_sk2_enhanced, s_seg2_enhanced = self.cross_interaction_h8(s_sk2, s_seg2)
        
        # 获取掩码语义向量
        cat_out = self.category_encoder(
            seg, 
            target_hw=(s_seg2.shape[2], s_seg2.shape[3]), 
            hard_assign=True
        )
        category_global = cat_out["global_semantic"] # [B, 128]
        probs = cat_out["probs"]                     # [B, H, W, K]
        
        # 🔥 计算空间动态掩码 Dynamic Mask (哪些区域需要运动)
        # 根据我们12类的定义：0:树 1:山 2:水 3:船 4:建筑 5:桥 6:石 7:人 8:云 9:花 10:动物 11:背景
        # 动态类/柔性类索引定义：2(水), 3(船), 7(人), 8(云), 10(动物), 11(未知留白区)
        dynamic_classes = [2, 3, 7, 8, 10, 11] 
        dynamic_mask = sum([probs[..., i] for i in dynamic_classes]).unsqueeze(1) # [B, 1, H, W]
        
        def flat(x):
            return rearrange(x, "b c h w -> b (h w) c")
            
        feat_h4 = torch.cat([flat(c1), flat(s_sk1_enhanced), flat(s_seg1_enhanced)], dim=-1)
        fused_h4 = self.fusion_h4(feat_h4)

        feat_h8 = torch.cat([flat(c2), flat(s_sk2_enhanced), flat(s_seg2_enhanced)], dim=-1)
        fused_h8 = self.fusion_h8(feat_h8)
        
        fused_structure = torch.cat([fused_h4, fused_h8], dim=1) # [B, L, _]
        
        L = fused_structure.shape[1]
        category_pooled = category_global.unsqueeze(1).expand(-1, L, -1)
        combined = torch.cat([fused_structure, category_pooled], dim=-1)
        
        structure_context = self.to_dit_context(combined)
        
        # 🔥 自适应预测时间流梯度曲线 [B, 81]
        time_decay_curve, dsrf_reg_loss = self.dsrf_layer(category_global, category_presence, dynamic_mask)
        self.dsrf_aux_loss = dsrf_reg_loss
        
        # 🔥 修改返回值，将动态掩码与推理出的时间曲线一并传递给 Pipeline
        if return_pyramid:
            return structure_context, dynamic_mask, time_decay_curve, {}
        return structure_context, dynamic_mask, time_decay_curve

def integrate_structure_encoder_to_pipeline(pipe, encoder: WanAdaptedDualCondEncoder, freeze_dit: bool = True, **kwargs):
    """
    将结构编码器集成到 WanVideoPipeline 中，并根据要求冻结主模型
    """
    # 挂载结构编码器
    pipe.structure_encoder = encoder.to(pipe.device, pipe.torch_dtype)
    
    # 冻结主网络的 DiT，因为只训练编码器
    if freeze_dit and hasattr(pipe, 'dit') and pipe.dit is not None:
        for param in pipe.dit.parameters():
            param.requires_grad = False
            
    # 也可以一并确保 VAE 和 Text Encoder 是冻结的 (通常流水线默认冻结，这里是为了双保险)
    if hasattr(pipe, 'vae') and pipe.vae is not None:
        for param in pipe.vae.parameters():
            param.requires_grad = False
    if hasattr(pipe, 'text_encoder') and pipe.text_encoder is not None:
        for param in pipe.text_encoder.parameters():
            param.requires_grad = False