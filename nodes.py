import os
import random
import re
import shutil
import folder_paths
import yaml
import numpy as np
import torch
import gc
import json
import hashlib
import comfy.model_management as mm
import comfy.utils
from typing import Tuple, Any
from comfy_api.latest import ComfyExtension, io

from .utility.download import download_from_huggingface, download_from_modelscope
from .glm_ocr import GlmOCR

glmOCR = GlmOCR()
fingerprint = 1

# 加载提示词模板
PROMPT_TEMPLATES_FILE = os.path.join(os.path.dirname(__file__), "prompt_templates.yaml")
with open(PROMPT_TEMPLATES_FILE, 'r', encoding='utf-8') as f:
    PROMPT_TEMPLATES_DATA = yaml.safe_load(f)
    PROMPT_TEMPLATE_NAMES = list(PROMPT_TEMPLATES_DATA.keys())

# 定义自定义类型
TYPE_GlmOCRModel = io.Custom(io_type="EASY_GLMOCR_MODEL")

# 下载并加载GLM-OCR模型节点
class DownloadAndLoadGlmOCRModel(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy downloadGlmOCRAndLoadModel",
            display_name="GLM-OCR Model Loader",
            category="EasyUse/GlmOCR",
            inputs=[
                io.Combo.Input("model_id", options=["zai-org/GLM-OCR"], default="zai-org/GLM-OCR"),
                io.Combo.Input("download_from", options=["huggingface", "modelscope"], default="huggingface"),
                io.Combo.Input("attention", options=["sdpa", "flash_attention_2", "eager"], default="sdpa"),
                io.AnyType.Input("start", optional=True, tooltip="Start to load or download models"),
            ],
            outputs=[
                TYPE_GlmOCRModel.Output(display_name="glm_ocr_model"),
            ],
            hidden=[io.Hidden.unique_id]
        )

    @classmethod
    async def fingerprint_inputs(cls, model_id: str, download_from: str, attention: str, start=None) -> str:
        global fingerprint
        base_fingerprint = f"{model_id}_{download_from}_{attention}"
        return f"{base_fingerprint}_{str(fingerprint)}"

    @classmethod
    def execute(cls, model_id: str, download_from: str, attention: str, start: io.AnyType = None) -> io.NodeOutput:
        llm_model_dir = os.path.join(folder_paths.models_dir, "LLM")
        if not os.path.exists(llm_model_dir):
            os.mkdir(llm_model_dir)

        model_name = os.path.basename(model_id).replace('.', '-')
        model_path = os.path.join(llm_model_dir, model_name)

        if not os.path.exists(model_path):
            print(f"Model not found locally. Downloading {model_name}...")
            if download_from == "huggingface":
                download_from_huggingface(model_id, model_path)
            elif download_from == "modelscope":
                download_from_modelscope(model_id, model_path)
            else:
                raise ValueError(f"Unsupported download source: {download_from}")
            print(f"Model downloaded to {model_path}...")

        model, processor = glmOCR.load_model(model_path, attn_implementation=attention)

        return io.NodeOutput({
            "model": model,
            "processor": processor
        })

# 应用GLM-OCR模型进行OCR识别
class ApplyGlmOCR(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="easy applyGlmOCR",
            display_name="Apply GLM-OCR",
            category="EasyUse/GlmOCR",
            inputs=[
                TYPE_GlmOCRModel.Input("glm_ocr_model"),
                io.Image.Input("image", display_name="Input Image"),
                io.Combo.Input("prompt_template", options=PROMPT_TEMPLATE_NAMES, default=PROMPT_TEMPLATE_NAMES[0], tooltip="prompt to guide the OCR process"),
                io.Boolean.Input("unload_model", label_on="on", label_off="off", default=False, tooltip="Unload model from VRAM after inference"),
                io.Float.Input("temperature", default=0.1, min=0.1, max=2.0, step=0.05),
                io.Int.Input("seed", default=0, min=0, max=2 ** 32 - 1),
                io.String.Input("custom_prompt", default="", force_input=True, optional=True, tooltip="Custom prompt template (optional, will override template if provided)"),
            ],
            outputs=[
                io.String.Output(display_name="result", tooltip="output the result"),
            ],
            hidden=[io.Hidden.unique_id]
        )

    @classmethod
    def execute(cls, glm_ocr_model: dict, image: Any, prompt_template: str, unload_model: bool, temperature: float, seed: int, custom_prompt: str = "") -> io.NodeOutput:
        if custom_prompt.strip():
            system_prompt = custom_prompt
        else:
            system_prompt = PROMPT_TEMPLATES_DATA[prompt_template]["system_prompt"]

        result = glmOCR.infer(glm_ocr_model, images=image, prompt=system_prompt, temperature=temperature, seed=seed)

        if isinstance(result, list) and len(result) > 0:
            result = result[0]

        # 卸载模型
        if unload_model:
            global fingerprint
            glmOCR.unload_model()
            fingerprint = random.randrange(100000, 999999)

        return io.NodeOutput(result)
