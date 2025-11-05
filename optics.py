"""
Camera Raw - 光学调整节点
"""

import torch
import numpy as np


class CameraRawOptics:
    """Camera Raw 光学调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "去除色差": ("BOOLEAN", {"default": False, "label_on": "启用", "label_off": "禁用"}),
                "去边数量": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "晕影数量": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "晕影中点": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "晕影圆度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "晕影羽化": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_optics"
    CATEGORY = "🔵BB camera raw"
    
    def apply_optics(self, image, 去除色差=False, 去边数量=0.0,
                    晕影数量=0.0, 晕影中点=50.0, 晕影圆度=0.0, 晕影羽化=50.0):
        """应用光学调整"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            
            if 去除色差 or 去边数量 > 0:
                img_array = self._apply_chromatic_aberration_removal(img_array, 去边数量)
            
            if 晕影数量 != 0:
                img_array = self._apply_vignette(img_array, 晕影数量, 晕影中点, 
                                                晕影圆度, 晕影羽化)
            
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_chromatic_aberration_removal(self, img, amount):
        """应用去色差"""
        try:
            from scipy import ndimage
            for c in range(3):
                img[:,:,c] = ndimage.gaussian_filter(img[:,:,c], sigma=amount / 100.0)
        except ImportError:
            pass
        
        return img
    
    def _apply_vignette(self, img, amount, midpoint, roundness, feather):
        """应用晕影效果"""
        h, w = img.shape[:2]
        center_x, center_y = w / 2, h / 2
        
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        max_dist = np.sqrt(center_x**2 + center_y**2)
        
        if roundness < 0:
            aspect = 1.0 + abs(roundness) / 100.0
            dist_from_center = np.sqrt(((x - center_x) / aspect)**2 + ((y - center_y) * aspect)**2)
        
        normalized_dist = dist_from_center / max_dist
        midpoint_factor = midpoint / 100.0
        normalized_dist = (normalized_dist - midpoint_factor) / (1.0 - midpoint_factor)
        normalized_dist = np.clip(normalized_dist, 0, 1)
        
        feather_factor = feather / 100.0
        vignette = 1.0 - normalized_dist * (1.0 - feather_factor)
        vignette = np.clip(vignette, 0, 1)
        
        vignette_factor = 1.0 - (amount / 100.0) * (1.0 - vignette)
        img = img * vignette_factor[..., np.newaxis]
        
        return img
