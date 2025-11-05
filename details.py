"""
Camera Raw - 细节调整节点
"""

import torch
import numpy as np
from PIL import Image, ImageFilter
from .utils import CameraRawUtils


class CameraRawDetails:
    """Camera Raw 细节调整节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "锐化数量": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 150.0, "step": 1.0}),
                "锐化半径": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "锐化细节": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "锐化蒙版": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色明亮度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色细节": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色对比度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色颜色": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色颜色细节": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "减少杂色颜色平滑度": ("FLOAT", {"default": 50.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_details"
    CATEGORY = "🔵BB camera raw"
    
    def apply_details(self, image, 锐化数量=0.0, 锐化半径=1.0, 
                     锐化细节=25.0, 锐化蒙版=0.0,
                     减少杂色明亮度=0.0, 减少杂色细节=50.0, 
                     减少杂色对比度=0.0, 减少杂色颜色=25.0, 
                     减少杂色颜色细节=50.0, 减少杂色颜色平滑度=50.0):
        """应用细节调整"""
        batch_size = image.shape[0]
        processed_images = []
        
        for i in range(batch_size):
            img = image[i].cpu().numpy()
            img_array = np.clip(img, 0.0, 1.0)
            
            if 锐化数量 > 0:
                img_array = self._apply_sharpening(img_array, 锐化数量, 锐化半径, 
                                                  锐化细节, 锐化蒙版)
            
            if 减少杂色明亮度 > 0 or 减少杂色颜色 > 0:
                img_array = self._apply_noise_reduction(img_array, 减少杂色明亮度, 
                                                      减少杂色细节, 减少杂色对比度,
                                                      减少杂色颜色, 减少杂色颜色细节,
                                                      减少杂色颜色平滑度)
            
            img_array = np.clip(img_array, 0.0, 1.0)
            processed_images.append(torch.from_numpy(img_array))
        
        return (torch.stack(processed_images),)
    
    def _apply_sharpening(self, img, amount, radius, detail, masking):
        """应用锐化"""
        try:
            from scipy import ndimage
            gray = np.mean(img, axis=2)
            laplacian = ndimage.laplace(gray)
            if masking > 0:
                edge_mask = np.abs(laplacian)
                edge_mask = np.clip(edge_mask * (masking / 100.0), 0, 1)
                sharpening = laplacian * (amount / 100.0) * edge_mask
            else:
                sharpening = laplacian * (amount / 100.0)
            
            sharpening_3d = sharpening[..., np.newaxis]
            img = img + sharpening_3d * (radius / 1.0)
        except ImportError:
            pil_img = Image.fromarray((img * 255).astype(np.uint8))
            if amount > 0:
                sharpened = pil_img.filter(ImageFilter.UnsharpMask(radius=radius, percent=int(amount), threshold=3))
                img = np.array(sharpened, dtype=np.float32) / 255.0
        
        return img
    
    def _apply_noise_reduction(self, img, luminance, detail, contrast, 
                               color, color_detail, color_smoothness):
        """应用降噪"""
        try:
            from scipy import ndimage
            if luminance > 0:
                img = ndimage.gaussian_filter(img, sigma=luminance / 50.0)
            
            if color > 0:
                hsv = CameraRawUtils.rgb_to_hsv(img)
                hsv[:,:,1] = ndimage.gaussian_filter(hsv[:,:,1], sigma=color / 50.0)
                img = CameraRawUtils.hsv_to_rgb(hsv)
        except ImportError:
            if luminance > 0:
                pil_img = Image.fromarray((img * 255).astype(np.uint8))
                blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=luminance / 20.0))
                img = np.array(blurred, dtype=np.float32) / 255.0
        
        return img

