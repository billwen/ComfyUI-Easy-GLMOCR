import numpy as np
from PIL import Image

def tensor_to_pil(image_tensor, batch_index=0) -> Image:
    # Convert tensor of shape [batch, height, width, channels] at the batch_index to PIL Image
    # 优化：直接处理tensor，减少内存操作
    img_tensor = image_tensor[batch_index]
    
    # 确保tensor在CPU上并转换为numpy
    if img_tensor.is_cuda:
        img_tensor = img_tensor.cpu()
    
    # 高效的数值转换
    img_np = (img_tensor * 255.0).clamp(0, 255).numpy().astype(np.uint8)
    
    # 处理维度
    if img_np.ndim == 3 and img_np.shape[-1] in [1, 3, 4]:
        if img_np.shape[-1] == 1:
            img_np = img_np.squeeze(-1)
    
    return Image.fromarray(img_np)