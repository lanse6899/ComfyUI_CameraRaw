"""
Camera Raw - 效果调整节点
"""

import torch
import numpy as np
from PIL import Image, ImageFilter


class CameraRawEffects:
    """Camera Raw 效果调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "纹理": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "清晰度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "去薄雾": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "🔵BB camera raw"
    
    def apply_effects(self, image, 纹理=0.0, 清晰度=0.0, 去薄雾=0.0):
        """应用效果调整"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            
            if 清晰度 != 0:
                img_array = self._apply_clarity(img_array, 清晰度)
            
            if 纹理 != 0:
                img_array = self._apply_texture(img_array, 纹理)
            
            if 去薄雾 != 0:
                img_array = self._apply_dehaze(img_array, 去薄雾)
            
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_clarity(self, img, clarity):
        """应用清晰度调整"""
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        midtone_mask = 1.0 - 4.0 * np.abs(gray - 0.5)
        midtone_mask = np.clip(midtone_mask, 0, 1)
        clarity_factor = clarity / 100.0
        contrast_adjustment = 1.0 + clarity_factor * midtone_mask[..., np.newaxis]
        img = 0.5 + (img - 0.5) * contrast_adjustment
        return img
    
    def _apply_texture(self, img, texture):
        """应用纹理调整"""
        try:
            from scipy import ndimage
            gray = np.mean(img, axis=2)
            blurred = ndimage.gaussian_filter(gray, sigma=2.0)
        except ImportError:
            gray = np.mean(img, axis=2)
            gray_pil = Image.fromarray((gray * 255).astype(np.uint8))
            blurred_pil = gray_pil.filter(ImageFilter.BLUR)
            blurred = np.array(blurred_pil, dtype=np.float32) / 255.0
        
        detail = gray - blurred
        texture_factor = texture / 100.0
        enhanced_detail = detail * texture_factor
        detail_3d = enhanced_detail[..., np.newaxis]
        img = img + detail_3d
        return img
    
    def _apply_dehaze(self, img, dehaze):
        """应用去薄雾效果"""
        dark_channel = np.min(img, axis=2)
        atmospheric_light = np.percentile(dark_channel, 99.9)
        dehaze_factor = dehaze / 100.0
        
        if dehaze > 0:
            transmission = 1.0 - dehaze_factor * (1.0 - dark_channel / (atmospheric_light + 1e-6))
            transmission = np.clip(transmission, 0.1, 1.0)
            img = (img - atmospheric_light) / transmission[..., np.newaxis] + atmospheric_light
        else:
            fog_strength = -dehaze_factor
            fog_color = np.ones_like(img) * atmospheric_light
            img = img * (1.0 - fog_strength) + fog_color * fog_strength
        
        return img
