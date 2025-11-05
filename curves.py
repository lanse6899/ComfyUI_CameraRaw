"""
Camera Raw - 曲线调整节点
"""

import torch
import numpy as np


class CameraRawCurves:
    """Camera Raw 曲线调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "曲线预设": (["线性", "中对比度", "强对比度", "自定义"], {
                    "default": "线性"
                }),
                "高光": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "亮部": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "暗部": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "阴影": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_curves"
    CATEGORY = "🔵BB camera raw"
    
    def apply_curves(self, image, 曲线预设="线性", 高光=0.0, 
                    亮部=0.0, 暗部=0.0, 阴影=0.0):
        """应用曲线调整"""
        # 根据预设设置默认曲线
        if 曲线预设 == "中对比度":
            高光, 亮部, 暗部, 阴影 = 10.0, 5.0, -5.0, -10.0
        elif 曲线预设 == "强对比度":
            高光, 亮部, 暗部, 阴影 = 20.0, 10.0, -10.0, -20.0
        
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            img_array = self._apply_curves(img_array, 曲线预设, 高光, 亮部, 暗部, 阴影)
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_curves(self, img, preset, highlights, lights, darks, shadows):
        """应用曲线调整"""
        gray = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        
        if highlights != 0:
            highlight_mask = np.maximum(gray - 0.75, 0) * 4
            highlight_adjust = (highlights / 100.0) * highlight_mask[..., np.newaxis]
            img = img + highlight_adjust
        
        if lights != 0:
            light_mask = np.maximum(np.minimum(gray - 0.5, 0.75 - gray), 0) * 4
            light_adjust = (lights / 100.0) * light_mask[..., np.newaxis]
            img = img + light_adjust
        
        if darks != 0:
            dark_mask = np.maximum(np.minimum(gray - 0.25, 0.5 - gray), 0) * 4
            dark_adjust = (darks / 100.0) * dark_mask[..., np.newaxis]
            img = img + dark_adjust
        
        if shadows != 0:
            shadow_mask = np.maximum(0.25 - gray, 0) * 4
            shadow_adjust = (shadows / 100.0) * shadow_mask[..., np.newaxis]
            img = img + shadow_adjust
        
        return img
