"""Wan-only slim model config placeholders.

This file keeps compatibility for modules that import these symbols,
while intentionally dropping non-Wan model registrations.
"""

from typing import Dict, List
from typing_extensions import Literal, TypeAlias

Preset_model_id: TypeAlias = Literal[
    "Wan2.2-TI2V-5B",
    "Wan2.2-T2V-A14B",
    "Wan2.2-I2V-A14B",
]

# The slim ModelManager uses its own internal WAN_MODEL_LOADER_CONFIGS.
model_loader_configs: List = []
huggingface_model_loader_configs: List = []
patch_model_loader_configs: List = []

# Kept for downloader interface compatibility.
preset_models_on_huggingface: Dict = {}
preset_models_on_modelscope: Dict = {}
