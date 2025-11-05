"""
Camera Raw - 颜色分级节点
"""

import torch
import numpy as np
from .utils import CameraRawUtils


class CameraRawColorGrading:
    """Camera Raw 颜色分级节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "阴影色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "阴影饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "阴影明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "中间调色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "中间调饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "中间调明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "高光色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "高光饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "高光明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_grading"
    CATEGORY = "🔵BB camera raw"
    
    def apply_color_grading(self, image, 阴影色相=0.0, 阴影饱和度=0.0, 阴影明度=0.0,
                           中间调色相=0.0, 中间调饱和度=0.0, 中间调明度=0.0,
                           高光色相=0.0, 高光饱和度=0.0, 高光明度=0.0):
        """应用颜色分级"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            img_array = self._apply_color_grading(img_array, 阴影色相, 阴影饱和度, 阴影明度,
                                                 中间调色相, 中间调饱和度, 中间调明度,
                                                 高光色相, 高光饱和度, 高光明度)
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_color_grading(self, img, sh_hue, sh_sat, sh_lum, 
                             mid_hue, mid_sat, mid_lum, 
                             hi_hue, hi_sat, hi_lum):
        """应用颜色分级"""
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        shadow_mask = np.maximum(0.33 - gray, 0) * 3
        shadow_mask = np.clip(shadow_mask, 0, 1)
        
        highlight_mask = np.maximum(gray - 0.67, 0) * 3
        highlight_mask = np.clip(highlight_mask, 0, 1)
        
        midtone_mask = 1.0 - shadow_mask - highlight_mask
        midtone_mask = np.clip(midtone_mask, 0, 1)
        
        hsv = CameraRawUtils.rgb_to_hsv(img)
        
        if sh_hue != 0 or sh_sat > 0 or sh_lum != 0:
            hsv[:,:,0] = (hsv[:,:,0] * 360 + sh_hue * shadow_mask[..., np.newaxis]) % 360 / 360.0
            hsv[:,:,1] = np.clip(hsv[:,:,1] + (sh_sat / 100.0) * shadow_mask[..., np.newaxis], 0, 1)
            hsv[:,:,2] = np.clip(hsv[:,:,2] + (sh_lum / 100.0) * shadow_mask[..., np.newaxis], 0, 1)
        
        if mid_hue != 0 or mid_sat > 0 or mid_lum != 0:
            hsv[:,:,0] = (hsv[:,:,0] * 360 + mid_hue * midtone_mask[..., np.newaxis]) % 360 / 360.0
            hsv[:,:,1] = np.clip(hsv[:,:,1] + (mid_sat / 100.0) * midtone_mask[..., np.newaxis], 0, 1)
            hsv[:,:,2] = np.clip(hsv[:,:,2] + (mid_lum / 100.0) * midtone_mask[..., np.newaxis], 0, 1)
        
        if hi_hue != 0 or hi_sat > 0 or hi_lum != 0:
            hsv[:,:,0] = (hsv[:,:,0] * 360 + hi_hue * highlight_mask[..., np.newaxis]) % 360 / 360.0
            hsv[:,:,1] = np.clip(hsv[:,:,1] + (hi_sat / 100.0) * highlight_mask[..., np.newaxis], 0, 1)
            hsv[:,:,2] = np.clip(hsv[:,:,2] + (hi_lum / 100.0) * highlight_mask[..., np.newaxis], 0, 1)
        
        img = CameraRawUtils.hsv_to_rgb(hsv)
        return img

