"""
Camera Raw 节点实现
每个功能对应一个独立的节点
"""

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageEnhance
import math
from scipy import ndimage


class CameraRawBrightness:
    """亮 - 亮度调整节点（曝光、对比度、高光、阴影、白色、黑色）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
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
    CATEGORY = "🔵BB CameraRaw"
    
    def apply_brightness(self, image, 曝光, 对比度, 高光, 阴影, 白色, 黑色):
        # 转换为numpy数组
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 曝光调整
        img = img * (2.0 ** 曝光)
        
        # 对比度调整
        contrast_factor = (100.0 + 对比度) / 100.0
        img = (img - 0.5) * contrast_factor + 0.5
        
        # 高光调整（S曲线）
        if 高光 != 0:
            highlight_factor = 高光 / 100.0
            mask = img > 0.5
            img[mask] = img[mask] + (1.0 - img[mask]) * highlight_factor * (img[mask] - 0.5) * 2
        
        # 阴影调整
        if 阴影 != 0:
            shadow_factor = 阴影 / 100.0
            mask = img < 0.5
            img[mask] = img[mask] + img[mask] * shadow_factor * (0.5 - img[mask]) * 2
        
        # 白色调整
        if 白色 != 0:
            white_factor = 白色 / 100.0
            img = img + (1.0 - img) * white_factor * img
        
        # 黑色调整
        if 黑色 != 0:
            black_factor = 黑色 / 100.0
            img = img - img * black_factor * (1.0 - img)
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawColor:
    """颜色 - 颜色调整节点（白平衡、温度、色调、自然饱和度、饱和度）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "温度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "色调": ("FLOAT", {"default": 0.0, "min": -150.0, "max": 150.0, "step": 1.0}),
                "自然饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color"
    CATEGORY = "🔵BB CameraRaw"
    
    def apply_color(self, image, 温度, 色调, 自然饱和度, 饱和度):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 温度调整（蓝色到黄色）
        temp_factor = 温度 / 100.0
        if temp_factor > 0:  # 变暖（增加黄色）
            img[:, :, :, 0] = np.clip(img[:, :, :, 0] + temp_factor * 0.1, 0, 1)  # R
            img[:, :, :, 2] = np.clip(img[:, :, :, 2] - temp_factor * 0.05, 0, 1)  # B
        else:  # 变冷（增加蓝色）
            img[:, :, :, 0] = np.clip(img[:, :, :, 0] + temp_factor * 0.1, 0, 1)  # R
            img[:, :, :, 2] = np.clip(img[:, :, :, 2] - temp_factor * 0.05, 0, 1)  # B
        
        # 色调调整（绿色到洋红）
        tint_factor = 色调 / 150.0
        img[:, :, :, 1] = np.clip(img[:, :, :, 1] - tint_factor * 0.05, 0, 1)  # G
        
        # 自然饱和度（Vibrance）- 只增强低饱和度区域
        if 自然饱和度 != 0:
            vibrance_factor = 1.0 + 自然饱和度 / 100.0
            gray = np.mean(img, axis=3, keepdims=True)
            saturation_mask = 1.0 - np.abs(img - gray) * 3.0
            saturation_mask = np.clip(saturation_mask, 0, 1)
            img = gray + (img - gray) * (1.0 + saturation_mask * (vibrance_factor - 1.0))
        
        # 饱和度调整
        if 饱和度 != 0:
            sat_factor = 1.0 + 饱和度 / 100.0
            gray = np.mean(img, axis=3, keepdims=True)
            img = gray + (img - gray) * sat_factor
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawEffects:
    """效果 - 效果调整节点（纹理、清晰度、去除薄雾、晕影、颗粒）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "纹理": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "清晰度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "去除薄雾": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "晕影": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "颗粒": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effects"
    CATEGORY = "🔵BB CameraRaw"
    
    def apply_effects(self, image, 纹理, 清晰度, 去除薄雾, 晕影, 颗粒):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        batch_size, height, width, channels = img.shape
        
        # 纹理调整（增强或减弱纹理细节）
        if 纹理 != 0:
            texture_factor = 纹理 / 100.0
            for i in range(batch_size):
                for c in range(channels):
                    channel = img[i, :, :, c].copy()
                    # 使用中值滤波和高斯模糊的差异来提取纹理
                    # 纹理增强：使用较大的模糊半径提取中频细节
                    blurred_large = ndimage.gaussian_filter(channel, sigma=3.0)
                    blurred_small = ndimage.gaussian_filter(channel, sigma=0.5)
                    texture_detail = blurred_small - blurred_large
                    # 应用纹理调整
                    img[i, :, :, c] = np.clip(channel + texture_detail * texture_factor * 0.5, 0, 1)
        
        # 清晰度调整（局部对比度增强，类似Camera Raw的Clarity）
        if 清晰度 != 0:
            clarity_factor = 清晰度 / 100.0
            for i in range(batch_size):
                for c in range(channels):
                    channel = img[i, :, :, c].copy()
                    # 使用Unsharp Masking方法（非锐化掩蔽）
                    # 这是Camera Raw中清晰度调整的标准方法
                    blurred = ndimage.gaussian_filter(channel, sigma=1.5)
                    # 计算细节（高频信息）
                    detail = channel - blurred
                    # 应用清晰度调整（局部对比度增强）
                    # 使用S曲线增强中间调对比度
                    enhanced = channel + detail * clarity_factor * 0.8
                    img[i, :, :, c] = np.clip(enhanced, 0, 1)
        
        # 去除薄雾调整
        if 去除薄雾 != 0:
            dehaze_factor = 去除薄雾 / 100.0
            for i in range(batch_size):
                # 计算暗通道
                dark_channel = np.min(img[i], axis=2)
                # 使用暗通道进行去雾
                atmospheric_light = np.percentile(dark_channel, 99)
                transmission = 1.0 - dehaze_factor * (dark_channel / (atmospheric_light + 1e-6))
                transmission = np.clip(transmission, 0.1, 1.0)
                transmission = np.expand_dims(transmission, axis=2)
                img[i] = (img[i] - atmospheric_light) / transmission + atmospheric_light
        
        # 晕影效果
        if 晕影 != 0:
            vignette_factor = 晕影 / 100.0
            center_y, center_x = height / 2, width / 2
            max_dist = math.sqrt(center_x**2 + center_y**2)
            
            y_coords, x_coords = np.ogrid[:height, :width]
            dist = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            vignette_mask = 1.0 - (dist / max_dist) * vignette_factor
            vignette_mask = np.clip(vignette_mask, 0, 1)
            vignette_mask = np.expand_dims(vignette_mask, axis=(0, 3))
            img = img * vignette_mask
        
        # 颗粒效果
        if 颗粒 > 0:
            grain_intensity = 颗粒 / 100.0
            noise = np.random.normal(0, grain_intensity * 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawColorMixer:
    """混色器 - HSL颜色混合器节点"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "红色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "红色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "橙色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "橙色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "橙色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黄色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "黄色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "黄色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "绿色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "青色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "青色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "青色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "蓝色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "紫色-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "紫色-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "紫色-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "洋红-色相": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 1.0}),
                "洋红-饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "洋红-亮度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_mixer"
    CATEGORY = "🔵BB CameraRaw"
    
    def rgb_to_hsl(self, rgb):
        """RGB转HSL"""
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        delta = max_val - min_val
        
        l = (max_val + min_val) / 2.0
        
        s = np.zeros_like(l)
        mask = delta != 0
        s[mask] = delta[mask] / (1.0 - np.abs(2.0 * l[mask] - 1.0) + 1e-6)
        
        h = np.zeros_like(l)
        mask_r = (max_val == r) & (delta != 0)
        mask_g = (max_val == g) & (delta != 0)
        mask_b = (max_val == b) & (delta != 0)
        
        h[mask_r] = 60.0 * (((g[mask_r] - b[mask_r]) / (delta[mask_r] + 1e-6)) % 6.0)
        h[mask_g] = 60.0 * (((b[mask_g] - r[mask_g]) / (delta[mask_g] + 1e-6)) + 2.0)
        h[mask_b] = 60.0 * (((r[mask_b] - g[mask_b]) / (delta[mask_b] + 1e-6)) + 4.0)
        h = h / 360.0
        
        return np.stack([h, s, l], axis=-1)
    
    def hsl_to_rgb(self, hsl):
        """HSL转RGB"""
        h, s, l = hsl[..., 0] * 360.0, hsl[..., 1], hsl[..., 2]
        
        c = (1.0 - np.abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - np.abs((h / 60.0) % 2.0 - 1.0))
        m = l - c / 2.0
        
        r = np.zeros_like(h)
        g = np.zeros_like(h)
        b = np.zeros_like(h)
        
        mask = (h >= 0) & (h < 60)
        r[mask], g[mask], b[mask] = c[mask], x[mask], 0
        
        mask = (h >= 60) & (h < 120)
        r[mask], g[mask], b[mask] = x[mask], c[mask], 0
        
        mask = (h >= 120) & (h < 180)
        r[mask], g[mask], b[mask] = 0, c[mask], x[mask]
        
        mask = (h >= 180) & (h < 240)
        r[mask], g[mask], b[mask] = 0, x[mask], c[mask]
        
        mask = (h >= 240) & (h < 300)
        r[mask], g[mask], b[mask] = x[mask], 0, c[mask]
        
        mask = (h >= 300) & (h < 360)
        r[mask], g[mask], b[mask] = c[mask], 0, x[mask]
        
        r, g, b = r + m, g + m, b + m
        return np.stack([r, g, b], axis=-1)
    
    def apply_color_mixer(self, image, **kwargs):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 颜色范围定义（HSL色相范围）
        color_ranges = {
            'red': (0, 15),
            'orange': (15, 45),
            'yellow': (45, 75),
            'green': (75, 150),
            'aqua': (150, 195),
            'blue': (195, 255),
            'purple': (255, 285),
            'magenta': (285, 345),
        }
        
        # 转换为HSL
        hsl = self.rgb_to_hsl(img)
        h = hsl[..., 0] * 360.0
        s = hsl[..., 1]
        l = hsl[..., 2]
        
        # 对每个颜色范围应用调整
        color_name_map = {
            'red': '红色',
            'orange': '橙色',
            'yellow': '黄色',
            'green': '绿色',
            'aqua': '青色',
            'blue': '蓝色',
            'purple': '紫色',
            'magenta': '洋红',
        }
        
        for color_name, (h_min, h_max) in color_ranges.items():
            chinese_name = color_name_map.get(color_name, color_name)
            hue_key = f"{chinese_name}-色相"
            sat_key = f"{chinese_name}-饱和度"
            lum_key = f"{chinese_name}-亮度"
            
            if hue_key not in kwargs:
                continue
            
            hue_adj = kwargs.get(hue_key, 0.0) / 180.0
            sat_adj = kwargs.get(sat_key, 0.0) / 100.0
            lum_adj = kwargs.get(lum_key, 0.0) / 100.0
            
            # 创建颜色范围掩码
            if h_min > h_max:  # 跨越0度的情况
                mask = (h >= h_min) | (h < h_max)
            else:
                mask = (h >= h_min) & (h < h_max)
            
            # 应用调整
            h[mask] = (h[mask] + hue_adj * 180.0) % 360.0
            s[mask] = np.clip(s[mask] + sat_adj, 0, 1)
            l[mask] = np.clip(l[mask] + lum_adj, 0, 1)
        
        # 转换回RGB
        hsl[..., 0] = h / 360.0
        hsl[..., 1] = s
        hsl[..., 2] = l
        img = self.hsl_to_rgb(hsl)
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawColorGrading:
    """颜色分级 - 分离色调节点（高光、中间调、阴影的色调调整）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "高光色相": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "高光饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "中间调色相": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "中间调饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "阴影色相": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "阴影饱和度": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "平衡": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_grading"
    CATEGORY = "🔵BB CameraRaw"
    
    def hsl_to_rgb_single(self, h, s, l):
        """单个HSL值转RGB"""
        h = h % 360.0
        c = (1.0 - abs(2.0 * l - 1.0)) * s
        x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
        m = l - c / 2.0
        
        if h < 60:
            r, g, b = c, x, 0
        elif h < 120:
            r, g, b = x, c, 0
        elif h < 180:
            r, g, b = 0, c, x
        elif h < 240:
            r, g, b = 0, x, c
        elif h < 300:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return r + m, g + m, b + m
    
    def apply_color_grading(self, image, 高光色相, 高光饱和度, 中间调色相, 中间调饱和度, 
                           阴影色相, 阴影饱和度, 平衡):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 计算亮度
        luminance = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        
        # 根据平衡调整高光和阴影的分界点
        balance_factor = 平衡 / 100.0
        highlight_threshold = 0.5 + balance_factor * 0.3
        shadow_threshold = 0.5 - balance_factor * 0.3
        
        # 创建掩码
        highlight_mask = np.clip((luminance - highlight_threshold) / (1.0 - highlight_threshold), 0, 1)
        shadow_mask = np.clip((shadow_threshold - luminance) / shadow_threshold, 0, 1)
        midtone_mask = 1.0 - highlight_mask - shadow_mask
        midtone_mask = np.clip(midtone_mask, 0, 1)
        
        # 将色调和饱和度转换为RGB偏移
        highlight_rgb = np.array(self.hsl_to_rgb_single(高光色相, 高光饱和度 / 100.0, 0.5))
        midtone_rgb = np.array(self.hsl_to_rgb_single(中间调色相, 中间调饱和度 / 100.0, 0.5))
        shadow_rgb = np.array(self.hsl_to_rgb_single(阴影色相, 阴影饱和度 / 100.0, 0.5))
        
        # 归一化到[-1, 1]范围
        highlight_rgb = (highlight_rgb - 0.5) * 2.0
        midtone_rgb = (midtone_rgb - 0.5) * 2.0
        shadow_rgb = (shadow_rgb - 0.5) * 2.0
        
        # 应用颜色分级
        for c in range(3):
            highlight_adj = highlight_rgb[c] * highlight_mask * 0.1
            midtone_adj = midtone_rgb[c] * midtone_mask * 0.1
            shadow_adj = shadow_rgb[c] * shadow_mask * 0.1
            img[..., c] = np.clip(img[..., c] + highlight_adj + midtone_adj + shadow_adj, 0, 1)
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawDetails:
    """细节 - 细节调整节点（锐化、降噪）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "锐化": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 150.0, "step": 1.0}),
                "半径": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 3.0, "step": 0.1}),
                "细节": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "蒙版": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "亮度降噪": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 1.0}),
                "颜色降噪": ("FLOAT", {"default": 25.0, "min": 0.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_details"
    CATEGORY = "🔵BB CameraRaw"
    
    def apply_details(self, image, 锐化, 半径, 细节, 蒙版, 亮度降噪, 颜色降噪):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        batch_size, height, width, channels = img.shape
        
        # 锐化处理
        if 锐化 > 0:
            for i in range(batch_size):
                for c in range(channels):
                    channel = img[i, :, :, c]
                    # 使用高斯模糊创建锐化掩码
                    blurred = ndimage.gaussian_filter(channel, sigma=半径)
                    sharp_mask = channel - blurred
                    
                    # 应用细节参数
                    detail_factor = 细节 / 100.0
                    sharp_mask = sharp_mask * detail_factor
                    
                    # 应用蒙版（保护平滑区域）
                    if 蒙版 > 0:
                        edge_strength = np.abs(sharp_mask)
                        mask_threshold = 蒙版 / 100.0
                        mask = edge_strength > mask_threshold
                        sharp_mask = sharp_mask * mask
                    
                    # 应用锐化
                    sharpening_factor = 锐化 / 150.0
                    img[i, :, :, c] = np.clip(channel + sharp_mask * sharpening_factor, 0, 1)
        
        # 亮度降噪
        if 亮度降噪 > 0:
            noise_sigma = 亮度降噪 / 100.0 * 0.05
            for i in range(batch_size):
                for c in range(channels):
                    img[i, :, :, c] = ndimage.gaussian_filter(img[i, :, :, c], sigma=noise_sigma * 10)
        
        # 颜色降噪
        if 颜色降噪 > 0:
            noise_sigma = 颜色降噪 / 100.0 * 0.02
            for i in range(batch_size):
                # 在颜色空间进行降噪
                lab = img[i]
                for c in range(channels):
                    lab[:, :, c] = ndimage.gaussian_filter(lab[:, :, c], sigma=noise_sigma * 10)
                img[i] = lab
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


class CameraRawCalibration:
    """校准 - 相机校准节点（RGB通道校准）"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "红色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "红色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "绿色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色色相": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
                "蓝色饱和度": ("FLOAT", {"default": 0.0, "min": -100.0, "max": 100.0, "step": 1.0}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_calibration"
    CATEGORY = "🔵BB CameraRaw"
    
    def apply_calibration(self, image, 红色色相, 红色饱和度, 绿色色相, 绿色饱和度, 蓝色色相, 蓝色饱和度):
        img = image.cpu().numpy()
        img = np.clip(img, 0, 1)
        
        # 红色通道校准
        if 红色色相 != 0 or 红色饱和度 != 0:
            hue_shift = 红色色相 / 100.0 * 0.1
            sat_factor = 1.0 + 红色饱和度 / 100.0
            gray = np.mean(img, axis=3, keepdims=True)
            red_channel = img[..., 0:1]
            red_channel = gray + (red_channel - gray) * sat_factor
            red_channel = np.clip(red_channel + hue_shift, 0, 1)
            img[..., 0] = red_channel[..., 0]
        
        # 绿色通道校准
        if 绿色色相 != 0 or 绿色饱和度 != 0:
            hue_shift = 绿色色相 / 100.0 * 0.1
            sat_factor = 1.0 + 绿色饱和度 / 100.0
            gray = np.mean(img, axis=3, keepdims=True)
            green_channel = img[..., 1:2]
            green_channel = gray + (green_channel - gray) * sat_factor
            green_channel = np.clip(green_channel + hue_shift, 0, 1)
            img[..., 1] = green_channel[..., 0]
        
        # 蓝色通道校准
        if 蓝色色相 != 0 or 蓝色饱和度 != 0:
            hue_shift = 蓝色色相 / 100.0 * 0.1
            sat_factor = 1.0 + 蓝色饱和度 / 100.0
            gray = np.mean(img, axis=3, keepdims=True)
            blue_channel = img[..., 2:3]
            blue_channel = gray + (blue_channel - gray) * sat_factor
            blue_channel = np.clip(blue_channel + hue_shift, 0, 1)
            img[..., 2] = blue_channel[..., 0]
        
        img = np.clip(img, 0, 1)
        return (torch.from_numpy(img),)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "CameraRawBrightness": CameraRawBrightness,
    "CameraRawColor": CameraRawColor,
    "CameraRawEffects": CameraRawEffects,
    "CameraRawColorMixer": CameraRawColorMixer,
    "CameraRawColorGrading": CameraRawColorGrading,
    "CameraRawDetails": CameraRawDetails,
    "CameraRawCalibration": CameraRawCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraRawBrightness": "🔵BB 亮 (Brightness)",
    "CameraRawColor": "🔵BB 颜色 (Color)",
    "CameraRawEffects": "🔵BB 效果 (Effects)",
    "CameraRawColorMixer": "🔵BB 混色器 (Color Mixer)",
    "CameraRawColorGrading": "🔵BB 颜色分级 (Color Grading)",
    "CameraRawDetails": "🔵BB 细节 (Details)",
    "CameraRawCalibration": "🔵BB 校准 (Calibration)",
}

