# ComfyUI Camera Raw 插件

一个完整的 Camera Raw 滤镜插件，包含 9 个独立的功能节点。

## 安装

1. 将整个 `ComfyUI_CameraRaw` 文件夹复制到 `ComfyUI/custom_nodes/` 目录
2. 重启 ComfyUI
3. 在节点菜单中找到 `🔵BB Camera Raw` 分类下的节点

## 节点列表

所有节点名称前都有 🔵BB 前缀标识：

1. **🔵BB Camera Raw - 亮度调整**
   - 曝光、对比度、高光、阴影、白色、黑色

2. **🔵BB Camera Raw - 颜色调整**
   - 白平衡预设、色温、色调、自然饱和度、饱和度

3. **🔵BB Camera Raw - 效果调整**
   - 纹理、清晰度、去薄雾

4. **🔵BB Camera Raw - 曲线调整**
   - 曲线预设、高光、亮部、暗部、阴影

5. **🔵BB Camera Raw - HSL 混色器**
   - 8 种颜色的色相、饱和度、明度调整

6. **🔵BB Camera Raw - 颜色分级**
   - 阴影、中间调、高光的分离色调

7. **🔵BB Camera Raw - 细节调整**
   - 锐化、降噪

8. **🔵BB Camera Raw - 光学调整**
   - 去色差、晕影

9. **🔵BB Camera Raw - 校准**
   - 阴影色调、原色调整

## 依赖

- 基础依赖：torch, numpy, PIL
- 可选依赖：scipy（用于更高质量的图像处理，如果没有会自动降级到 PIL 实现）

## 许可证

MIT License


