# 模型源（Model Source）

`eyegaze.onnx` 是 **EyeGaze 模型的完整导出**（Qualcomm AI Hub 的 `eyegaze` 模型，
由官方发布的 `.pt` 权重导出）。

它在本项目中**不参与推理**，而是 RKNN 转换链的起点：

```
assets/model_src/eyegaze.onnx        本文件（完整模型，含 softargmax 与 FC 头）
        │  python -m split_model
        ▼
assets/eyegaze_backbone.onnx          主干（卷积 + Hourglass），可被 RKNN 编译
        │  .venv-rknn/bin/python export_rknn.py
        ▼
assets/eyegaze_rk3576.rknn            RK3576 NPU 模型（实际推理用）
```

之所以要先切分：完整模型里的 **softargmax 关键点解码**（对 48×80 空间位置做
softmax 再加权求和）和 **7748 输入的全连接 gaze 头**都无法在 NPU 上表达，因此拆成
「主干上 NPU + 头留 CPU」，见 `../README.md`。

重新生成主干（若需改动切分方式）：

```bash
python -m split_model --verify
```
