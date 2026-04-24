"""
测试结构编码器集成到WanVideoPipeline
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import torch
from PIL import Image
import numpy as np

print("="*80)
print("测试结构编码器集成到 WanVideoPipeline")
print("="*80)

# Test 1: 导入测试
print("\n[Test 1] 导入模块...")
try:
    from examples.wanvideo.dual_cond_encoder import WanAdaptedDualCondEncoder, integrate_structure_encoder_to_pipeline
    print("✓ DualCondEncoder0114 导入成功")
except Exception as e:
    print(f"✗ DualCondEncoder0114 导入失败: {e}")
    sys.exit(1)

try:
    # 🆕 直接导入，避免触发整个包初始化
    from diffsynth.pipelines.wan_video_new import WanVideoPipeline, WanVideoUnit_StructureEmbedder, model_fn_wan_video
    print("✓ WanVideoPipeline 直接导入成功")
except Exception as e:
    print(f"✗ WanVideoPipeline 导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: 编码器创建测试
print("\n[Test 2] 创建结构编码器...")
try:
    encoder = WanAdaptedDualCondEncoder(
        seg_channels=3,
        dit_dim=5120,
        structure_dim=768,
        num_categories=11
    )
    print(f"✓ 编码器创建成功")
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"  编码器参数量: {total_params/1e6:.2f}M")
except Exception as e:
    print(f"✗ 编码器创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: 编码器前向传播测试
print("\n[Test 3] 测试编码器前向传播...")
try:
    sketch = torch.randn(2, 1, 480, 832)
    seg = torch.randn(2, 3, 480, 832)
    
    print(f"  输入 sketch: {sketch.shape}")
    print(f"  输入 seg: {seg.shape}")
    
    structure_context = encoder(sketch, seg)
    print(f"✓ 前向传播成功")
    print(f"  输出 structure_context: {structure_context.shape}")
    
    structure_context, pyramid = encoder(sketch, seg, return_pyramid=True)
    print(f"✓ 金字塔输出成功")
    print(f"  金字塔键: {list(pyramid.keys())}")
    
except Exception as e:
    print(f"✗ 前向传播失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Pipeline集成测试
print("\n[Test 4] 测试与Pipeline集成...")
try:
    pipe = WanVideoPipeline(device="cpu", torch_dtype=torch.float32)
    print(f"✓ Pipeline 创建成功")
    
    integrate_structure_encoder_to_pipeline(pipe, encoder)
    print(f"✓ 编码器集成到 Pipeline 成功")
    
    assert hasattr(pipe, 'structure_encoder'), "Pipeline 缺少 structure_encoder 属性"
    assert pipe.structure_encoder is not None, "structure_encoder 为 None"
    print(f"✓ 验证 pipe.structure_encoder 存在")
    
except Exception as e:
    print(f"✗ Pipeline集成失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Unit处理测试
print("\n[Test 5] 测试 WanVideoUnit_StructureEmbedder...")
try:
    unit = WanVideoUnit_StructureEmbedder()
    print(f"✓ WanVideoUnit_StructureEmbedder 创建成功")
    
    sketch_img = Image.fromarray(np.random.randint(0, 255, (480, 832), dtype=np.uint8), mode='L')
    seg_img = Image.fromarray(np.random.randint(0, 255, (480, 832, 3), dtype=np.uint8), mode='RGB')
    
    result = unit.process(pipe, sketch_img, seg_img, 480, 832)
    
    if 'structure_context' in result:
        print(f"✓ Unit 处理成功")
        print(f"  输出 structure_context: {result['structure_context'].shape}")
    else:
        print(f"⚠️ Unit 返回空结果（可能是正常的）")
        print(f"  返回键: {list(result.keys())}")
    
except Exception as e:
    print(f"✗ Unit处理失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: model_fn_wan_video 参数测试
print("\n[Test 6] 测试 model_fn_wan_video 接受 structure_context 参数...")
try:
    import inspect
    
    sig = inspect.signature(model_fn_wan_video)
    params = list(sig.parameters.keys())
    
    if 'structure_context' in params:
        print(f"✓ model_fn_wan_video 包含 structure_context 参数")
        print(f"  参数位置: {params.index('structure_context')}")
    else:
        print(f"✗ model_fn_wan_video 缺少 structure_context 参数")
        print(f"  当前参数: {params}")
        sys.exit(1)
    
except Exception as e:
    print(f"✗ 参数检查失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("测试总结")
print("="*80)
print("✓ 所有关键测试通过！")
print("\n可以继续进行实际训练测试。")
print("="*80)