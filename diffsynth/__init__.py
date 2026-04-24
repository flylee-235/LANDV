"""Slim diffsynth package for Wan-only training/inference."""

import importlib

__all__ = []


def _safe_import(module_name: str):
    try:
        mod = importlib.import_module(f".{module_name}", package=__name__)
        globals()[module_name] = mod
        __all__.append(module_name)
    except Exception as e:
        print(f"[WARN] skip importing diffsynth.{module_name}: {e}")


for _m in ["utils", "models", "pipelines", "prompters", "schedulers", "trainers", "lora", "vram_management", "configs"]:
    _safe_import(_m)
