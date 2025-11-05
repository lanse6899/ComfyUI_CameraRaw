"""
Camera Raw - 亮度调整节点
"""

import torch
import numpy as np
from PIL import Image, ImageEnhance


class CameraRawBrightness:
    """Camera Raw 亮度调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "曝光": ("FLOAT", {"default": 0.0, "min": -5.0, "max": 5.0, "step": 0.1}),
                "对比度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "高光": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "阴影": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "白色": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黑色": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_brightness"
    CATEGORY = "🔵BB camera raw"
    
    def apply_brightness(self, image, 曝光=0.0, 对比度=0.0, 高光=0.0, 
                        阴影=0.0, 白色=0.0, 黑色=0.0):
        """应用亮度调整"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img = (img * 255.0).astype(np.uint8)
            pil_image = Image.fromarray(img)
            
            if 曝光 != 0:
                exposure_factor = 2 ** 曝光
                pil_image = ImageEnhance.Brightness(pil_image).enhance(exposure_factor)
            
            if 对比度 != 0:
                contrast_factor = 1.0 + (对比度 / 100.0)
                pil_image = ImageEnhance.Contrast(pil_image).enhance(contrast_factor)
            
            img_array = np.array(pil_image, dtype=np.float32) / 255.0
            
            if 高光 != 0 or 阴影 != 0:
                img_array = self._adjust_highlights_shadows(img_array, 高光, 阴影)
            
            if 白色 != 0 or 黑色 != 0:
                img_array = self._adjust_whites_blacks(img_array, 白色, 黑色)
            
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _adjust_highlights_shadows(self, img, highlights, shadows):
        """调整高光和阴影"""
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        if highlights != 0:
            highlight_mask = np.maximum(gray - 0.5, 0) * 2
            highlight_factor = 1.0 + (highlights / 100.0) * highlight_mask[..., np.newaxis]
            img = img + (img * highlight_factor - img) * highlight_mask[..., np.newaxis]
        
        if shadows != 0:
            shadow_mask = np.maximum(0.5 - gray, 0) * 2
            shadow_factor = 1.0 + (shadows / 100.0) * shadow_mask[..., np.newaxis]
            img = img * (1 - shadow_mask[..., np.newaxis]) + img * shadow_factor * shadow_mask[..., np.newaxis]
        
        return img
    
    def _adjust_whites_blacks(self, img, whites, blacks):
        """调整白色和黑色"""
        if whites != 0:
            white_factor = 1.0 + (whites / 100.0)
            img = np.power(img, 1.0 / white_factor)
        
        if blacks != 0:
            black_factor = 1.0 + (blacks / 100.0)
            img = np.power(img, black_factor)
        
        return img
