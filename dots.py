import os
import shutil
import time
import torch
import folder_paths
from typing import Any, Optional
from transformers.configuration_utils import PretrainedConfig
from transformers.models.qwen2 import Qwen2Config
from transformers import Qwen2_5_VLProcessor, AutoProcessor
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from .utility.image import tensor_to_pil

class DotsVisionConfig(PretrainedConfig):
    model_type: str = "dots_vit"

    def __init__(
        self,
        embed_dim: int = 1536,  # vision encoder embed size
        hidden_size: int = 1536,  # after merger hidden size
        intermediate_size: int = 4224,
        num_hidden_layers: int = 42,
        num_attention_heads: int = 12,
        num_channels: int = 3,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        temporal_patch_size: int = 1,
        rms_norm_eps: float = 1e-5,
        use_bias: bool = False,
        attn_implementation="flash_attention_2",  # "eager", "sdpa", "flash_attention_2"
        initializer_range=0.02,
        init_merger_std=0.02,
        is_causal=False,  # ve causal forward
        post_norm=True,
        gradient_checkpointing=False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.rms_norm_eps = rms_norm_eps
        self.use_bias = use_bias
        self.attn_implementation = attn_implementation
        self.initializer_range = initializer_range
        self.init_merger_std = init_merger_std
        self.is_causal = is_causal
        self.post_norm = post_norm
        self.gradient_checkpointing = gradient_checkpointing


class DotsOCRConfig(Qwen2Config):
    model_type = "dots_ocr"
    def __init__(self,
        image_token_id = 151665,
        video_token_id = 151656,
        vision_config: Optional[dict] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_config = DotsVisionConfig(**(vision_config or {}))

    def save_pretrained(self, save_directory, **kwargs):
        self._auto_class = None
        super().save_pretrained(save_directory, **kwargs)


class DotsVLProcessor(Qwen2_5_VLProcessor):

    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kwargs):
        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.image_token = "<|imgpad|>" if not hasattr(tokenizer, "image_token") else tokenizer.image_token
        self.image_token_id = 151665 if not hasattr(tokenizer, "image_token_id") else tokenizer.image_token_id

class DotsOCR:
    def __init__(self):
        self.model = None
        self.processor = None
        self.attn_implementation = "flash_attention_2"

    def load_model(self, model_path, attn_implementation: str = "flash_attention_2"):
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.attn_implementation = attn_implementation
        print('Loading model with attention implementation:', attn_implementation)

        if self.model is None or self.attn_implementation != attn_implementation:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            try:
                if hasattr(torch, 'compile') and torch.cuda.is_available():
                    model = torch.compile(model, mode="reduce-overhead")
                    print("enable torch.compile")
            except Exception as e:
                print(f"torch.compile failed，using origin model: {e}")
            self.model = model
        else:
            model = self.model

        if self.processor is None:
            processor = DotsVLProcessor.from_pretrained(
                model_path,
                attn_implementation=attn_implementation,
                trust_remote_code=True
            )
        else:
            processor = self.processor

        return model, processor
    
    def unload_model(self):
        del self.processor
        del self.model
        self.processor = None
        self.model = None
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    def infer(self, ocr_models, images, prompt, seed=-1, max_new_tokens=2048, do_sample=False, temperature=0.1, top_p=0.9, num_beams=1, enable_timing_print=False):
        
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("Please install qwen_vl_utils: pip install qwen-vl-utils")
        
        timing_info = {}
        start_time = time.time()

        self.model = ocr_models["model"]
        self.processor = ocr_models["processor"]
        
        # 设置随机种子（如果需要）
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # 优化图像转换，避免不必要的内存拷贝
        with torch.no_grad():
            if enable_timing_print:
                t1 = time.time()
            pil_image = tensor_to_pil(images)
            if enable_timing_print:
                timing_info['image_conversion'] = time.time() - t1
                    
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": pil_image
                        },
                        {"type": "text", "text": prompt}
                    ]
                }
            ]

            # 准备推理输入
            if enable_timing_print:
                t2 = time.time()

            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda", non_blocking=True)
            
            if enable_timing_print:
                timing_info['preprocessing'] = time.time() - t2

            # 优化生成参数，提升推理速度
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                "use_cache": True,  # 启用KV缓存
                "num_beams": num_beams,
            }
            
            # 推理生成输出
            if enable_timing_print:
                torch.cuda.synchronize()
                t3 = time.time()
            generated_ids = self.model.generate(**inputs, **generation_config)
            if enable_timing_print:
                torch.cuda.synchronize()
                timing_info['model_inference'] = time.time() - t3
            
            # 高效的token处理
            if enable_timing_print:
                t4 = time.time()
            input_length = inputs.input_ids.shape[1]
            generated_ids_trimmed = generated_ids[:, input_length:]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )
            if enable_timing_print:
                timing_info['postprocessing'] = time.time() - t4
                timing_info['total_time'] = time.time() - start_time
                print("性能分析:")
                for key, value in timing_info.items():
                    print(f"  {key}: {value:.3f}秒")

        
        return output_text