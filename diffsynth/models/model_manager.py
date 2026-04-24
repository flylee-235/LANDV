import importlib
import json
import os
from typing import List

import torch

from .utils import (
    hash_state_dict_keys,
    init_weights_on_device,
    load_state_dict,
    split_state_dict_with_prefix,
)
from .wan_video_animate_adapter import WanAnimateAdapter
from .wan_video_dit import WanModel
from .wan_video_dit_s2v import WanS2VModel
from .wan_video_image_encoder import WanImageEncoder
from .wan_video_motion_controller import WanMotionControllerModel
from .wan_video_text_encoder import WanTextEncoder
from .wan_video_vace import VaceWanModel
from .wan_video_vae import WanVideoVAE, WanVideoVAE38
from .wav2vec import WanS2VAudioEncoder


def load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        print(f"    model_name: {model_name} model_class: {model_class.__name__}")
        state_dict_converter = model_class.state_dict_converter()
        if model_resource == "civitai":
            state_dict_results = state_dict_converter.from_civitai(state_dict)
        elif model_resource == "diffusers":
            state_dict_results = state_dict_converter.from_diffusers(state_dict)
        else:
            raise ValueError(f"Unsupported model resource: {model_resource}")

        if isinstance(state_dict_results, tuple):
            model_state_dict, extra_kwargs = state_dict_results
            print(f"        This model is initialized with extra kwargs: {extra_kwargs}")
        else:
            model_state_dict, extra_kwargs = state_dict_results, {}

        model_torch_dtype = torch.float32 if extra_kwargs.get("upcast_to_float32", False) else torch_dtype
        with init_weights_on_device():
            model = model_class(**extra_kwargs)
        if hasattr(model, "eval"):
            model = model.eval()
        model.load_state_dict(model_state_dict, assign=True)
        model = model.to(dtype=model_torch_dtype, device=device)

        loaded_model_names.append(model_name)
        loaded_models.append(model)

    return loaded_model_names, loaded_models


def load_model_from_huggingface_folder(file_path, model_names, model_classes, torch_dtype, device):
    loaded_model_names, loaded_models = [], []
    for model_name, model_class in zip(model_names, model_classes):
        if torch_dtype in [torch.float32, torch.float16, torch.bfloat16]:
            model = model_class.from_pretrained(file_path, torch_dtype=torch_dtype).eval()
        else:
            model = model_class.from_pretrained(file_path).eval().to(dtype=torch_dtype)
        if torch_dtype == torch.float16 and hasattr(model, "half"):
            model = model.half()
        try:
            model = model.to(device=device)
        except Exception:
            pass
        loaded_model_names.append(model_name)
        loaded_models.append(model)
    return loaded_model_names, loaded_models


# Wan-only detection metadata (copied from original full config)
# format: (keys_hash_with_shape, [model_names], [model_classes], model_resource)
WAN_MODEL_LOADER_CONFIGS = [
    ("9269f8db9040a9d860eaca435be61814", ["wan_video_dit"], [WanModel], "civitai"),
    ("aafcfd9672c3a2456dc46e1cb6e52c70", ["wan_video_dit"], [WanModel], "civitai"),
    ("6bfcfb3b342cb286ce886889d519a77e", ["wan_video_dit"], [WanModel], "civitai"),
    ("6d6ccde6845b95ad9114ab993d917893", ["wan_video_dit"], [WanModel], "civitai"),
    ("349723183fc063b2bfc10bb2835cf677", ["wan_video_dit"], [WanModel], "civitai"),
    ("efa44cddf936c70abd0ea28b6cbe946c", ["wan_video_dit"], [WanModel], "civitai"),
    ("3ef3b1f8e1dab83d5b71fd7b617f859f", ["wan_video_dit"], [WanModel], "civitai"),
    ("70ddad9d3a133785da5ea371aae09504", ["wan_video_dit"], [WanModel], "civitai"),
    ("26bde73488a92e64cc20b0a7485b9e5b", ["wan_video_dit"], [WanModel], "civitai"),
    ("ac6a5aa74f4a0aab6f64eb9a72f19901", ["wan_video_dit"], [WanModel], "civitai"),
    ("b61c605c2adbd23124d152ed28e049ae", ["wan_video_dit"], [WanModel], "civitai"),
    ("1f5ab7703c6fc803fdded85ff040c316", ["wan_video_dit"], [WanModel], "civitai"),
    ("5b013604280dd715f8457c6ed6d6a626", ["wan_video_dit"], [WanModel], "civitai"),
    ("2267d489f0ceb9f21836532952852ee5", ["wan_video_dit"], [WanModel], "civitai"),
    ("47dbeab5e560db3180adf51dc0232fb1", ["wan_video_dit"], [WanModel], "civitai"),
    ("cb104773c6c2cb6df4f9529ad5c60d0b", ["wan_video_dit"], [WanModel], "diffusers"),
    ("966cffdcc52f9c46c391768b27637614", ["wan_video_dit"], [WanS2VModel], "civitai"),
    ("a61453409b67cd3246cf0c3bebad47ba", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    ("7a513e1f257a861512b1afd387a8ecd9", ["wan_video_dit", "wan_video_vace"], [WanModel, VaceWanModel], "civitai"),
    ("9c8818c2cbea55eca56c7b447df170da", ["wan_video_text_encoder"], [WanTextEncoder], "civitai"),
    ("5941c53e207d62f20f9025686193c40b", ["wan_video_image_encoder"], [WanImageEncoder], "civitai"),
    ("1378ea763357eea97acdef78e65d6d96", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    ("ccc42284ea13e1ad04693284c7a09be6", ["wan_video_vae"], [WanVideoVAE], "civitai"),
    ("e1de6c02cdac79f8b739f4d3698cd216", ["wan_video_vae"], [WanVideoVAE38], "civitai"),
    ("dbd5ec76bbf977983f972c151d545389", ["wan_video_motion_controller"], [WanMotionControllerModel], "civitai"),
    ("06be60f3a4526586d8431cd038a71486", ["wans2v_audio_encoder"], [WanS2VAudioEncoder], "civitai"),
    ("31fa352acb8a1b1d33cd8764273d80a2", ["wan_video_dit", "wan_video_animate_adapter"], [WanModel, WanAnimateAdapter], "civitai"),
]


# Optional lightweight HuggingFace folder detection.
# format: (architecture, huggingface_lib, model_name, redirected_architecture)
WAN_HF_LOADER_CONFIGS = [
    ("T5EncoderModel", "transformers.models.t5.modeling_t5", "wan_video_text_encoder", None),
]


class ModelDetectorFromSingleFile:
    def __init__(self, model_loader_configs):
        self.keys_hash_with_shape_dict = {}
        for keys_hash_with_shape, model_names, model_classes, model_resource in model_loader_configs:
            self.keys_hash_with_shape_dict[keys_hash_with_shape] = (model_names, model_classes, model_resource)

    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        return keys_hash_with_shape in self.keys_hash_with_shape_dict

    def load(self, file_path="", state_dict=None, device="cuda", torch_dtype=torch.float16, allowed_model_names=None, **kwargs):
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        keys_hash_with_shape = hash_state_dict_keys(state_dict, with_shape=True)
        if keys_hash_with_shape not in self.keys_hash_with_shape_dict:
            return [], []

        model_names, model_classes, model_resource = self.keys_hash_with_shape_dict[keys_hash_with_shape]
        if allowed_model_names:
            pairs = [(n, c) for n, c in zip(model_names, model_classes) if n in set(allowed_model_names)]
            if len(pairs) == 0:
                return [], []
            model_names = [p[0] for p in pairs]
            model_classes = [p[1] for p in pairs]
        return load_model_from_single_file(state_dict, model_names, model_classes, model_resource, torch_dtype, device)


class ModelDetectorFromSplitedSingleFile(ModelDetectorFromSingleFile):
    def match(self, file_path="", state_dict=None):
        if isinstance(file_path, str) and os.path.isdir(file_path):
            return False
        if state_dict is None:
            state_dict = load_state_dict(file_path)
        split_state_dict = split_state_dict_with_prefix(state_dict)
        return any(super().match(file_path, sub_state) for sub_state in split_state_dict)

    def load(self, file_path="", state_dict=None, device="cuda", torch_dtype=torch.float16, allowed_model_names=None, **kwargs):
        if state_dict is None:
            state_dict = load_state_dict(file_path)

        loaded_model_names, loaded_models = [], []
        split_state_dict = split_state_dict_with_prefix(state_dict)
        for sub_state in split_state_dict:
            if super().match(file_path, sub_state):
                names, models = super().load(
                    file_path,
                    sub_state,
                    device=device,
                    torch_dtype=torch_dtype,
                    allowed_model_names=allowed_model_names,
                )
                loaded_model_names.extend(names)
                loaded_models.extend(models)
        return loaded_model_names, loaded_models


class ModelDetectorFromHuggingfaceFolder:
    def __init__(self, model_loader_configs):
        self.architecture_dict = {}
        for architecture, huggingface_lib, model_name, redirected_architecture in model_loader_configs:
            self.architecture_dict[architecture] = (huggingface_lib, model_name, redirected_architecture)

    def match(self, file_path="", state_dict=None):
        if not isinstance(file_path, str) or os.path.isfile(file_path):
            return False
        config_path = os.path.join(file_path, "config.json")
        if not os.path.exists(config_path):
            return False
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return "architectures" in config or "_class_name" in config

    def load(self, file_path="", state_dict=None, device="cuda", torch_dtype=torch.float16, allowed_model_names=None, **kwargs):
        with open(os.path.join(file_path, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        loaded_model_names, loaded_models = [], []
        architectures = config.get("architectures", [config.get("_class_name")])
        for architecture in architectures:
            if architecture not in self.architecture_dict:
                continue
            huggingface_lib, model_name, redirected_architecture = self.architecture_dict[architecture]
            if allowed_model_names and model_name not in set(allowed_model_names):
                continue
            model_arch = redirected_architecture or architecture
            model_class = importlib.import_module(huggingface_lib).__getattribute__(model_arch)
            names, models = load_model_from_huggingface_folder(file_path, [model_name], [model_class], torch_dtype, device)
            loaded_model_names.extend(names)
            loaded_models.extend(models)
        return loaded_model_names, loaded_models


class ModelManager:
    def __init__(
        self,
        torch_dtype=torch.float16,
        device="cuda",
        model_id_list: List[str] = None,
        downloading_priority: List[str] = None,
        file_path_list: List[str] = None,
    ):
        if model_id_list:
            print("[WARN] model_id_list download is not enabled in Wan-only slim ModelManager. Please pass local file paths.")
        self.torch_dtype = torch_dtype
        self.device = device
        self.model = []
        self.model_path = []
        self.model_name = []
        self.model_detector = [
            ModelDetectorFromSingleFile(WAN_MODEL_LOADER_CONFIGS),
            ModelDetectorFromSplitedSingleFile(WAN_MODEL_LOADER_CONFIGS),
            ModelDetectorFromHuggingfaceFolder(WAN_HF_LOADER_CONFIGS),
        ]
        if file_path_list:
            self.load_models(file_path_list)

    def load_model(self, file_path, model_names=None, device=None, torch_dtype=None):
        print(f"Loading models from: {file_path}")
        if device is None:
            device = self.device
        if torch_dtype is None:
            torch_dtype = self.torch_dtype

        if isinstance(file_path, list):
            state_dict = {}
            for path in file_path:
                state_dict.update(load_state_dict(path))
        elif os.path.isfile(file_path):
            state_dict = load_state_dict(file_path)
        else:
            state_dict = None

        for model_detector in self.model_detector:
            if model_detector.match(file_path, state_dict):
                loaded_model_names, loaded_models = model_detector.load(
                    file_path,
                    state_dict,
                    device=device,
                    torch_dtype=torch_dtype,
                    allowed_model_names=model_names,
                    model_manager=self,
                )
                for loaded_name, loaded_model in zip(loaded_model_names, loaded_models):
                    self.model.append(loaded_model)
                    self.model_path.append(file_path)
                    self.model_name.append(loaded_name)
                print(f"    The following models are loaded: {loaded_model_names}.")
                break
        else:
            print("    We cannot detect the model type. No models are loaded.")

    def load_models(self, file_path_list, model_names=None, device=None, torch_dtype=None):
        for file_path in file_path_list:
            self.load_model(file_path, model_names=model_names, device=device, torch_dtype=torch_dtype)

    def fetch_model(self, model_name, file_path=None, require_model_path=False, index=None):
        fetched_models = []
        fetched_model_paths = []
        for model, model_path, model_name_ in zip(self.model, self.model_path, self.model_name):
            if file_path is not None and file_path != model_path:
                continue
            if model_name == model_name_:
                fetched_models.append(model)
                fetched_model_paths.append(model_path)

        if len(fetched_models) == 0:
            print(f"No {model_name} models available.")
            return None

        if len(fetched_models) == 1:
            print(f"Using {model_name} from {fetched_model_paths[0]}.")
            model = fetched_models[0]
            path = fetched_model_paths[0]
        else:
            if index is None:
                model = fetched_models[0]
                path = fetched_model_paths[0]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using {fetched_model_paths[0]}.")
            elif isinstance(index, int):
                model = fetched_models[:index]
                path = fetched_model_paths[:index]
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using first {index} entries.")
            else:
                model = fetched_models
                path = fetched_model_paths
                print(f"More than one {model_name} models are loaded: {fetched_model_paths}. Using all entries.")

        if require_model_path:
            return model, path
        return model

    def to(self, device):
        for model in self.model:
            model.to(device)
