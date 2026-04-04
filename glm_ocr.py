import time
import torch
from typing import Any
from .utility.image import tensor_to_pil


class GlmOCR:
    def __init__(self):
        self.model = None
        self.processor = None
        self.attn_implementation = "sdpa"

    def load_model(self, model_path, attn_implementation: str = "sdpa"):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.attn_implementation = attn_implementation
        print('Loading GLM-OCR model with attention implementation:', attn_implementation)

        if self.model is None or self.attn_implementation != attn_implementation:
            model = AutoModelForImageTextToText.from_pretrained(
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
                print(f"torch.compile failed, using origin model: {e}")
            self.model = model
        else:
            model = self.model

        if self.processor is None:
            processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            self.processor = processor
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

    def infer(self, ocr_models, images, prompt, seed=-1, max_new_tokens=8192, do_sample=False, temperature=0.1, top_p=0.9, num_beams=1, enable_timing_print=False):

        timing_info = {}
        start_time = time.time()

        self.model = ocr_models["model"]
        self.processor = ocr_models["processor"]

        # Set random seed if needed
        if seed != -1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        with torch.no_grad():
            if enable_timing_print:
                t1 = time.time()
            pil_image = tensor_to_pil(images)
            if enable_timing_print:
                timing_info['image_conversion'] = time.time() - t1

            # Build messages in the format GLM-OCR expects
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

            # Prepare inference inputs using apply_chat_template
            if enable_timing_print:
                t2 = time.time()

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            # Remove token_type_ids if present (GLM-OCR specific)
            inputs.pop("token_type_ids", None)

            if enable_timing_print:
                timing_info['preprocessing'] = time.time() - t2

            # Generation config
            generation_config = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "temperature": temperature,
                "use_cache": True,
                "num_beams": num_beams,
            }

            # Run inference
            if enable_timing_print:
                torch.cuda.synchronize()
                t3 = time.time()
            generated_ids = self.model.generate(**inputs, **generation_config)
            if enable_timing_print:
                torch.cuda.synchronize()
                timing_info['model_inference'] = time.time() - t3

            # Decode output - trim input tokens
            if enable_timing_print:
                t4 = time.time()
            input_length = inputs["input_ids"].shape[1]
            generated_ids_trimmed = generated_ids[:, input_length:]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            if enable_timing_print:
                timing_info['postprocessing'] = time.time() - t4
                timing_info['total_time'] = time.time() - start_time
                print("Performance analysis:")
                for key, value in timing_info.items():
                    print(f"  {key}: {value:.3f}s")

        return output_text
