"""
Camera Raw - 校准节点
"""

import torch
import numpy as np
from .utils import CameraRawUtils


class CameraRawCalibration:
    """Camera Raw 校准节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "阴影色调": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红原色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红原色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿原色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿原色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝原色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝原色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_calibration"
    CATEGORY = "🔵BB camera raw"
    
    def apply_calibration(self, image, 阴影色调=0.0, 红原色色相=0.0, 红原色饱和度=0.0,
                         绿原色色相=0.0, 绿原色饱和度=0.0, 蓝原色色相=0.0, 蓝原色饱和度=0.0):
        """应用校准"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            img_array = self._apply_calibration(img_array, 阴影色调, 红原色色相, 红原色饱和度, 
                                             绿原色色相, 绿原色饱和度, 蓝原色色相, 蓝原色饱和度)
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_calibration(self, img, shadow_tint, r_hue, r_sat, g_hue, g_sat, b_hue, b_sat):
        """应用校准"""
        hsv = CameraRawUtils.rgb_to_hsv(img)
        
        if shadow_tint != 0:
            gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
            shadow_mask = np.maximum(0.5 - gray, 0) * 2
            hsv[:,:,0] = (hsv[:,:,0] * 360 + shadow_tint * shadow_mask[..., np.newaxis]) % 360 / 360.0
        
        img = CameraRawUtils.hsv_to_rgb(hsv)
        return img
