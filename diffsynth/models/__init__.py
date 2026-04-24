# diffsynth/models/__init__.py
# from .model_manager import *  # 注释掉这行，避免触发整个导入链

# 只导入 Wan 系列需要的模块
from .wan_video_dit import WanModel, RMSNorm, sinusoidal_embedding_1d
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vae import WanVideoVAE, WanVideoVAE38
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_vace import VaceWanModel
from .wan_video_motion_controller import WanMotionControllerModel
from .wan_video_animate_adapter import WanAnimateAdapter
from .wan_video_dit_s2v import WanS2VModel
from .model_manager import ModelManager, load_state_dict

# 删除这行！现在编码器在 examples/wanvideo/
# from .DualCondEncoder0114 import WanAdaptedDualCondEncoder, integrate_structure_encoder_to_pipeline

__all__ = [
    'WanModel', 'RMSNorm', 'sinusoidal_embedding_1d',
    'WanTextEncoder',
    'WanVideoVAE', 'WanVideoVAE38',
    'WanImageEncoder',
    'VaceWanModel',
    'WanMotionControllerModel',
    'WanAnimateAdapter',
    'WanS2VModel',
    'ModelManager', 'load_state_dict',
]