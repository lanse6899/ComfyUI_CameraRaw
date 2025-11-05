"""
ComfyUI Camera Raw 插件
包含 9 个独立的 Camera Raw 功能节点
"""

from .brightness import CameraRawBrightness
from .color import CameraRawColor
from .effects import CameraRawEffects
from .curves import CameraRawCurves
from .hsl import CameraRawHSL
from .color_grading import CameraRawColorGrading
from .details import CameraRawDetails
from .optics import CameraRawOptics
from .calibration import CameraRawCalibration

# 节点映射
NODE_CLASS_MAPPINGS = {
    "CameraRawBrightness": CameraRawBrightness,
    "CameraRawColor": CameraRawColor,
    "CameraRawEffects": CameraRawEffects,
    "CameraRawCurves": CameraRawCurves,
    "CameraRawHSL": CameraRawHSL,
    "CameraRawColorGrading": CameraRawColorGrading,
    "CameraRawDetails": CameraRawDetails,
    "CameraRawOptics": CameraRawOptics,
    "CameraRawCalibration": CameraRawCalibration,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CameraRawBrightness": "🔵BB Camera Raw - 亮度调整",
    "CameraRawColor": "🔵BB Camera Raw - 颜色调整",
    "CameraRawEffects": "🔵BB Camera Raw - 效果调整",
    "CameraRawCurves": "🔵BB Camera Raw - 曲线调整",
    "CameraRawHSL": "🔵BB Camera Raw - HSL 混色器",
    "CameraRawColorGrading": "🔵BB Camera Raw - 颜色分级",
    "CameraRawDetails": "🔵BB Camera Raw - 细节调整",
    "CameraRawOptics": "🔵BB Camera Raw - 光学调整",
    "CameraRawCalibration": "🔵BB Camera Raw - 校准",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

