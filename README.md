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

## Acknowledgements

- [GLM-OCR](https://github.com/zai-org/GLM-OCR) by Z.ai for the OCR model
- [ComfyUI-Easy-DotsOCR](https://github.com/yolain/ComfyUI-Easy-DotsOCR) by yolain for the original node implementation
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) for the node workflow framework

## Contact

For issues or feature requests, please open an issue on this repository.
