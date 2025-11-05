"""
Camera Raw - 颜色调整节点
"""

import torch
import numpy as np


class CameraRawColor:
    """Camera Raw 颜色调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "白平衡预设": (["原照设置", "自动", "日光", "阴天", "阴影", "钨丝灯", "荧光灯", "闪光灯", "自定义"], {
                    "default": "原照设置"
                }),
                "色温": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "色调": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "自然饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color"
    CATEGORY = "🔵BB camera raw"
    
    def apply_color(self, image, 白平衡预设="原照设置", 色温=0.0, 
                   色调=0.0, 自然饱和度=0.0, 饱和度=0.0):
        """应用颜色调整"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            
            # 应用白平衡预设
            if 白平衡预设 != "原照设置" and 白平衡预设 != "自定义":
                temp, tin = self._get_white_balance_preset(白平衡预设)
                if 色温 == 0.0:
                    色温 = temp
                if 色调 == 0.0:
                    色调 = tin
            
            # 应用色温和色调
            if 色温 != 0.0 or 色调 != 0.0:
                img_array = self._apply_white_balance(img_array, 色温, 色调)
            
            # 应用自然饱和度和饱和度
            if 自然饱和度 != 0 or 饱和度 != 0:
                img_array = self._adjust_saturation(img_array, 自然饱和度, 饱和度)
            
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _get_white_balance_preset(self, preset):
        """获取白平衡预设值"""
        presets = {
            "自动": (0.0, 0.0),
            "日光": (0.0, 0.0),
            "阴天": (10.0, 0.0),
            "阴影": (20.0, 0.0),
            "钨丝灯": (-50.0, 0.0),
            "荧光灯": (-30.0, 20.0),
            "闪光灯": (0.0, 0.0),
        }
        return presets.get(preset, (0.0, 0.0))
    
    def _apply_white_balance(self, img, temperature, tint):
        """应用白平衡"""
        if temperature != 0:
            temp_factor = temperature / 100.0
            if temp_factor > 0:
                img[:,:,2] = np.clip(img[:,:,2] - temp_factor * 0.1, 0, 1)
                img[:,:,0] = np.clip(img[:,:,0] + temp_factor * 0.05, 0, 1)
                img[:,:,1] = np.clip(img[:,:,1] + temp_factor * 0.05, 0, 1)
            else:
                img[:,:,2] = np.clip(img[:,:,2] - temp_factor * 0.1, 0, 1)
                img[:,:,0] = np.clip(img[:,:,0] + temp_factor * 0.05, 0, 1)
                img[:,:,1] = np.clip(img[:,:,1] + temp_factor * 0.05, 0, 1)
        
        if tint != 0:
            tint_factor = tint / 150.0
            img[:,:,1] = np.clip(img[:,:,1] - tint_factor * 0.05, 0, 1)
            img[:,:,0] = np.clip(img[:,:,0] + tint_factor * 0.05, 0, 1)
        
        return img
    
    def _adjust_saturation(self, img, vibrance, saturation):
        """调整自然饱和度和饱和度"""
        max_channel = np.max(img, axis=2)
        min_channel = np.min(img, axis=2)
        delta = max_channel - min_channel
        current_saturation = np.where(max_channel > 0, delta / (max_channel + 1e-6), 0)
        
        if vibrance != 0:
            vibrance_factor = 1.0 + (vibrance / 100.0) * (1.0 - current_saturation)
            for c in range(3):
                img[:,:,c] = img[:,:,c] + (img[:,:,c] - max_channel) * (vibrance_factor - 1.0) * 0.5
        
        if saturation != 0:
            saturation_factor = 1.0 + (saturation / 100.0)
            for c in range(3):
                img[:,:,c] = max_channel + (img[:,:,c] - max_channel) * saturation_factor
        
        return img
