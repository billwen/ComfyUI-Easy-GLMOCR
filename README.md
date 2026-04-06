# ComfyUI-Easy-GLMOCR

ComfyUI-Easy-GLMOCR is a custom node for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) that provides text extraction via the [GLM-OCR](https://github.com/zai-org/GLM-OCR) engine. GLM-OCR is a tiny 0.9B-parameter vision-language model that delivers state-of-the-art OCR performance across text, formula, and table recognition.

## Features

- **Text Recognition** — Extract plain text from images
- **Formula Recognition** — Convert mathematical formulas to LaTeX
- **Table Recognition** — Parse tables into structured HTML
- **Layout Analysis** — Detect document layout elements with bounding boxes
- **Lightweight** — Only 0.9B parameters, runs efficiently on consumer GPUs
- **Auto Download** — Models downloaded automatically from HuggingFace or ModelScope

## Installation
-1. Check Python and CUDA version
   ```bash
   python --version
   # Python 3.14.3

   python -c "import torch; print(torch.__version__);print(torch.cuda.is_available())"
   # 2.11.0+cu130
   # True

   nvcc -V
   # nvcc: NVIDIA (R) Cuda compiler driver
   # Copyright (c) 2005-2026 NVIDIA Corporation
   # Built on Mon_Mar_02_09:52:23_PM_PST_2026
   # Cuda compilation tools, release 13.2, V13.2.51
   # Build cuda_13.2.r13.2/compiler.37434383_0
   ```
   * PyTorch's CUDA version maste match the nvcc's CUDA version.
   * on Ubuntu 24.04, use "sudo update-alternatives --config cuda" select the activated cuda version.

0. Install flash-attn
FlashAttention is very heavy to compile (lots of CUDA kernels) and needs massive amount of memory.
   ```bash
   # on 64GB machine, maximum parallel jobs are 4
   export MAX_JOBS=4 
   pip install flash-attn --no-build-isolation
   ```

   to add expand swap file
   ```bash
   sudo fallocate -l 64G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

1. Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   git clone https://github.com/billwen/ComfyUI-Easy-GLMOCR
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Restart ComfyUI. The GLM-OCR nodes will be available in the node list under **EasyUse/GlmOCR**.

## Nodes

### GLM-OCR Model Loader

Loads the GLM-OCR model. Downloads automatically on first use.

| Input | Description |
|---|---|
| `model_id` | Model identifier (`zai-org/GLM-OCR`) |
| `download_from` | Download source: `huggingface` or `modelscope` |
| `attention` | Attention implementation: `sdpa` (default), `flash_attention_2`, `eager` |

### Apply GLM-OCR

Runs OCR inference on an input image.

| Input | Description |
|---|---|
| `glm_ocr_model` | Model from the loader node |
| `image` | Input image |
| `prompt_template` | Built-in prompt (text/formula/table recognition, layout analysis, etc.) |
| `temperature` | Generation temperature (0.1 - 2.0) |
| `seed` | Random seed for reproducibility |
| `unload_model` | Free VRAM after inference |
| `custom_prompt` | Optional custom prompt (overrides template) |

## Usage

1. Add **GLM-OCR Model Loader** node to your workflow.
2. Add **Apply GLM-OCR** node and connect the model output.
3. Connect an image input to the Apply node.
4. Select a prompt template (e.g., `prompt_text_recognition`).
5. Run the workflow to extract text from the image.

## Download Models (Manual)

If you prefer to download models manually:

Download [all model files](https://huggingface.co/zai-org/GLM-OCR/tree/main) to `ComfyUI/models/LLM/GLM-OCR`

## File Structure

- `glm_ocr.py`: Core GLM-OCR model wrapper and inference logic
- `nodes.py`: ComfyUI node definitions
- `utility/`: Helper functions for image conversion and model downloading
- `prompt_templates.yaml`: Prompt templates for OCR tasks
- `requirements.txt`: Python dependencies

## Development

This project is developed based on [ComfyUI-Easy-DotsOCR](https://github.com/yolain/ComfyUI-Easy-DotsOCR) and adapted for the [GLM-OCR](https://github.com/zai-org/GLM-OCR) model by Z.ai.

## License

This project inherits the license from [GLM-OCR](https://github.com/zai-org/GLM-OCR). Please review their license terms before use.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

This project was forked and adapted from [ComfyUI-Easy-DotsOCR](https://github.com/yolain/ComfyUI-Easy-DotsOCR) by [yolain](https://github.com/yolain). The original project provided the ComfyUI node architecture, model loading/downloading pipeline, prompt template system, and image processing utilities that serve as the foundation of this project. We gratefully acknowledge yolain's work in building the original DotsOCR integration for ComfyUI.

## Acknowledgements

- [ComfyUI-Easy-DotsOCR](https://github.com/yolain/ComfyUI-Easy-DotsOCR) by [yolain](https://github.com/yolain) — Original project providing the node framework, download pipeline, and overall architecture
- [GLM-OCR](https://github.com/zai-org/GLM-OCR) by [Z.ai](https://github.com/zai-org) — The 0.9B vision-language OCR model powering text, formula, and table recognition
- [dots.ocr](https://github.com/rednote-hilab/dots.ocr) by [RedNote HILab](https://github.com/rednote-hilab) — The OCR model supported by the original upstream project
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by [comfyanonymous](https://github.com/comfyanonymous) — The node-based workflow framework

## Contact

For issues or feature requests, please open an issue on this repository.
