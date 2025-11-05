"""
Camera Raw 工具类
包含共享的辅助方法
"""

import numpy as np


class CameraRawUtils:
    """Camera Raw 工具类，包含共享的辅助方法"""
    
    @staticmethod
    def rgb_to_hsv(rgb):
        """RGB 转 HSV"""
        hsv = np.zeros_like(rgb)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        
        max_val = np.max(rgb, axis=2)
        min_val = np.min(rgb, axis=2)
        delta = max_val - min_val
        
        h = np.zeros_like(max_val)
        mask = delta != 0
        h[mask & (max_val == r)] = ((g - b) / delta)[mask & (max_val == r)] % 6
        h[mask & (max_val == g)] = ((b - r) / delta + 2)[mask & (max_val == g)]
        h[mask & (max_val == b)] = ((r - g) / delta + 4)[mask & (max_val == b)]
        hsv[:,:,0] = h / 6.0
        
        hsv[:,:,1] = np.where(max_val > 0, delta / max_val, 0)
        hsv[:,:,2] = max_val
        
        return hsv
    
    @staticmethod
    def hsv_to_rgb(hsv):
        """HSV 转 RGB"""
        h, s, v = hsv[:,:,0] * 6, hsv[:,:,1], hsv[:,:,2]
        
        i = np.floor(h).astype(np.int32) % 6
        f = h - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        rgb = np.zeros_like(hsv)
        rgb[:,:,0] = np.where(i == 0, v, np.where(i == 1, q, np.where(i == 2, p, np.where(i == 3, p, np.where(i == 4, t, v)))))
        rgb[:,:,1] = np.where(i == 0, t, np.where(i == 1, v, np.where(i == 2, v, np.where(i == 3, q, np.where(i == 4, p, p)))))
        rgb[:,:,2] = np.where(i == 0, p, np.where(i == 1, p, np.where(i == 2, t, np.where(i == 3, v, np.where(i == 4, v, q)))))
        
        return rgb

