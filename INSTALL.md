# 安装指南

## 快速安装

1. **复制插件文件夹**
   - 将整个 `ComfyUI_CameraRaw` 文件夹复制到 ComfyUI 的 `custom_nodes` 目录
   - 路径通常是：`ComfyUI/custom_nodes/ComfyUI_CameraRaw/`

2. **安装依赖**
   ```bash
   cd ComfyUI
   pip install -r custom_nodes/ComfyUI_CameraRaw/requirements.txt
   ```

3. **重启 ComfyUI**
   - 关闭 ComfyUI（如果正在运行）
   - 重新启动 ComfyUI
   - 插件会自动加载

4. **验证安装**
   - 在 ComfyUI 界面中，右键点击空白处
   - 选择 "Add Node" → "CameraRaw"
   - 应该能看到10个节点：
     - 亮 (Brightness)
     - 颜色 (Color)
     - 效果 (Effects)
     - 曲线 (Curves)
     - 混色器 (Color Mixer)
     - 颜色分级 (Color Grading)
     - 细节 (Details)
     - 光学 (Optics)
     - 镜头模糊 (Lens Blur)
     - 校准 (Calibration)

## 故障排除

### 如果节点没有出现

1. 检查文件夹结构是否正确
2. 检查 `__init__.py` 文件是否存在
3. 查看 ComfyUI 控制台是否有错误信息
4. 确认依赖已正确安装

### 如果出现导入错误

1. 确认已安装所有依赖：
   ```bash
   pip install numpy pillow scipy
   ```

2. 检查 Python 版本（建议 Python 3.8+）

### 如果节点运行缓慢

- 某些节点（如镜头模糊、畸变校正）计算量较大
- 对于大图像，处理可能需要一些时间
- 这是正常现象

## 使用建议

1. **节点串联**：可以将多个节点串联使用，实现复杂的图像调整流程
2. **参数调整**：建议从默认值开始，逐步调整参数
3. **性能优化**：对于批量处理，建议先调整参数，再应用到整个批次

