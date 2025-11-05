"""
Camera Raw - HSL 混色器节点
"""

import torch
import numpy as np
from .utils import CameraRawUtils


class CameraRawHSL:
    """Camera Raw HSL 混色器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "红色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "橙色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "橙色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "橙色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黄色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黄色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黄色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "浅绿色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "浅绿色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "浅绿色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "紫色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "紫色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "紫色明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "洋红色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "洋红饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "洋红明度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_hsl"
    CATEGORY = "🔵BB camera raw"
    
    def apply_hsl(self, image, 红色色相=0.0, 红色饱和度=0.0, 红色明度=0.0,
                 橙色色相=0.0, 橙色饱和度=0.0, 橙色明度=0.0,
                 黄色色相=0.0, 黄色饱和度=0.0, 黄色明度=0.0,
                 绿色色相=0.0, 绿色饱和度=0.0, 绿色明度=0.0,
                 浅绿色色相=0.0, 浅绿色饱和度=0.0, 浅绿色明度=0.0,
                 蓝色色相=0.0, 蓝色饱和度=0.0, 蓝色明度=0.0,
                 紫色色相=0.0, 紫色饱和度=0.0, 紫色明度=0.0,
                 洋红色相=0.0, 洋红饱和度=0.0, 洋红明度=0.0):
        """应用 HSL 混色器"""
        hsl_params = {
            'red': (红色色相, 红色饱和度, 红色明度),
            'orange': (橙色色相, 橙色饱和度, 橙色明度),
            'yellow': (黄色色相, 黄色饱和度, 黄色明度),
            'green': (绿色色相, 绿色饱和度, 绿色明度),
            'aqua': (浅绿色色相, 浅绿色饱和度, 浅绿色明度),
            'blue': (蓝色色相, 蓝色饱和度, 蓝色明度),
            'purple': (紫色色相, 紫色饱和度, 紫色明度),
            'magenta': (洋红色相, 洋红饱和度, 洋红明度),
        }
        
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            img_array = self._apply_hsl_mixer(img_array, hsl_params)
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_hsl_mixer(self, img, hsl_params):
        """应用 HSL 混色器"""
        hsv = CameraRawUtils.rgb_to_hsv(img)
        
        color_ranges = {
            'red': (0, 15, 345, 360),
            'orange': (15, 45),
            'yellow': (45, 75),
            'green': (75, 165),
            'aqua': (165, 195),
            'blue': (195, 255),
            'purple': (255, 285),
            'magenta': (285, 345),
        }
        
        h, s, v = hsv[:,:,0] * 360, hsv[:,:,1], hsv[:,:,2]
        
        for color_name, (hue, sat, lum) in hsl_params.items():
            if hue == 0 and sat == 0 and lum == 0:
                continue
            
            if color_name in color_ranges:
                range_def = color_ranges[color_name]
                if len(range_def) == 2:
                    mask = (h >= range_def[0]) & (h < range_def[1])
                else:
                    mask = ((h >= range_def[0]) & (h <= range_def[1])) | ((h >= range_def[2]) & (h <= range_def[3]))
                
                if hue != 0:
                    h[mask] = h[mask] + hue
                    h[mask] = h[mask] % 360
                
                if sat != 0:
                    s[mask] = np.clip(s[mask] + sat / 100.0, 0, 1)
                
                if lum != 0:
                    v[mask] = np.clip(v[mask] + lum / 100.0, 0, 1)
        
        hsv[:,:,0], hsv[:,:,1], hsv[:,:,2] = h / 360.0, s, v
        img = CameraRawUtils.hsv_to_rgb(hsv)
        
        return img

