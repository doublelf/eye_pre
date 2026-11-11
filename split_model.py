"""Split the exported EyeGaze ONNX model into an accelerator friendly backbone.

This is step 1 of the RKNN pipeline::

    assets/model_src/eyegaze.onnx   (full model)
        -> assets/eyegaze_backbone.onnx   (this script)
        -> assets/eyegaze_rk3576.rknn     (export_rknn.py, on the NPU)

The full model (``assets/model_src/eyegaze.onnx``) contains two parts:

* a convolutional backbone (``pre`` -> hourglass stacks -> ``pre2``) whose
  output is the 34 landmark heatmaps plus the 7680 element gaze feature,
* a decoding head (softargmax + two fully connected layers) that turns
  those tensors into a pitch/yaw pair.

Only the backbone is compiled for the NPU; the head stays on the host
(see :mod:`gaze_head`).  The produced backbone graph has two outputs::

    heatmaps : (batch, 34, 48, 80)   float32
    gaze_feat: (batch, 7680)         float32

Usage
-----
    python -m split_model                      # writes assets/eyegaze_backbone.onnx
    python -m split_model --verify              # + numerical parity check
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
ROOT = HERE
DEFAULT_MODEL = ROOT / "assets" / "model_src" / "eyegaze.onnx"
DEFAULT_BACKBONE = ROOT / "assets" / "eyegaze_backbone.onnx"
DEFAULT_HEAD = ROOT / "assets" / "eyegaze_head.npz"

# Name of the last hourglass stack's landmark convolution output.
LAST_STACK_CONV = "/outs.2/conv/Conv_output_0"


def _producers(graph: onnx.GraphProto) -> dict[str, onnx.NodeProto]:
    return {out: node for node in graph.node for out in node.output}


def _reachable(
    graph: onnx.GraphProto, tensor_names: list[str]
) -> tuple[set[int], set[str]]:
    """Nodes (by index) and tensors needed to produce ``tensor_names``."""
    producers = _producers(graph)
    init_names = {init.name for init in graph.initializer}
    graph_inputs = {i.name for i in graph.input}

    keep_nodes: set[int] = set()
    keep_tensors: set[str] = set()
    stack = list(tensor_names)
    while stack:
        name = stack.pop()
        if name in keep_tensors:
            continue
        keep_tensors.add(name)
        node = producers.get(name)
        if node is None:
            continue
        idx = next(i for i, n in enumerate(graph.node) if n is node)
        keep_nodes.add(idx)
        for inp in node.input:
            if inp and inp not in init_names and inp not in graph_inputs:
                stack.append(inp)
    return keep_nodes, keep_tensors


def split(
    model_path: Path = DEFAULT_MODEL,
    backbone_path: Path = DEFAULT_BACKBONE,
    head_path: Path = DEFAULT_HEAD,
) -> tuple[Path, Path]:
    """Write the backbone ONNX and the host-side head weights."""
    model = onnx.load(str(model_path))
    graph = model.graph

    producers = _producers(graph)
    if LAST_STACK_CONV not in producers:
        raise RuntimeError(
            f"{LAST_STACK_CONV} not found - the exported model layout changed?"
        )

    # gaze_feat: the Flatten node that consumes the pre2 output.
    gaze_feat_tensor = None
    for node in graph.node:
        if node.op_type == "Flatten" and node.output[0] == "/Flatten_output_0":
            gaze_feat_tensor = node.output[0]
            break
    if gaze_feat_tensor is None:
        raise RuntimeError("gaze feature tensor (/Flatten_output_0) not found")

    keep_nodes, keep_tensors = _reachable(graph, [LAST_STACK_CONV, gaze_feat_tensor])

    new_nodes = [node for i, node in enumerate(graph.node) if i in keep_nodes]

    # The exported graph takes a 3D image (N, 96, 160) and unsqueezes it
    # internally.  Accelerators want a plain 4D NCHW input, so the
    # unsqueeze is folded into the graph input.
    for node in list(new_nodes):
        if node.op_type == "Unsqueeze" and node.input and node.input[0] == "image":
            for consumer in new_nodes:
                for i, inp in enumerate(consumer.input):
                    if inp == node.output[0]:
                        consumer.input[i] = "image"
            new_nodes.remove(node)
            break

    # Give the two graph outputs friendly names.
    renames = {LAST_STACK_CONV: "heatmaps", gaze_feat_tensor: "gaze_feat"}
    for node in new_nodes:
        for i, out in enumerate(node.output):
            if out in renames:
                node.output[i] = renames[out]
        for i, inp in enumerate(node.input):
            if inp in renames:
                node.input[i] = renames[inp]

    used_initializers = {
        init.name
        for init in graph.initializer
        if init.name in keep_tensors
        or any(init.name in node.input for node in new_nodes)
    }

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name="eyegaze_backbone",
        inputs=[
            helper.make_tensor_value_info(
                "image", TensorProto.FLOAT, ["batch", 1, 96, 160]
            )
        ],
        outputs=[
            helper.make_tensor_value_info(
                "heatmaps", TensorProto.FLOAT, ["batch", 34, 48, 80]
            ),
            helper.make_tensor_value_info(
                "gaze_feat", TensorProto.FLOAT, ["batch", 7680]
            ),
        ],
        initializer=[
            init for init in graph.initializer if init.name in used_initializers
        ],
    )

    new_model = helper.make_model(
        new_graph,
        producer_name="eyegaze-split",
        opset_imports=model.opset_import,
    )
    new_model.ir_version = model.ir_version
    onnx.checker.check_model(new_model)

    backbone_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(new_model, str(backbone_path))

    # --- host side head weights ---
    initializers = {i.name: numpy_helper.to_array(i) for i in graph.initializer}
    head = {
        "fc1_weight": initializers["gaze_fc1.weight"],
        "fc1_bias": initializers["gaze_fc1.bias"],
        "fc2_weight": initializers["gaze_fc2.weight"],
        "fc2_bias": initializers["gaze_fc2.bias"],
    }
    head_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(head_path), **head)

    print(f"backbone : {backbone_path}")
    print(f"  nodes  : {len(new_nodes)} / {len(graph.node)}")
    print(f"  inputs : image (batch, 1, 96, 160) float32")
    print(f"  outputs: heatmaps (batch, 34, 48, 80), gaze_feat (batch, 7680)")
    print(f"head     : {head_path}")
    print(f"  fc1    : {head['fc1_weight'].shape}  fc2: {head['fc2_weight'].shape}")
    return backbone_path, head_path


def verify(
    model_path: Path = DEFAULT_MODEL,
    backbone_path: Path = DEFAULT_BACKBONE,
    head_path: Path = DEFAULT_HEAD,
    sample: Path | None = None,
) -> None:
    """Check that backbone + host head reproduces the original model output."""
    import onnxruntime as ort

    from gaze_head import GazeHead, softargmax2d
    from preprocess import preprocess_eye_crop

    sample = sample or (ROOT / "assets" / "input_image.png")
    image = preprocess_eye_crop(_read_gray(sample), "left")
    batch = image[None].astype(np.float32)

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    reference = session.run(None, {"image": batch})

    backbone = ort.InferenceSession(
        str(backbone_path), providers=["CPUExecutionProvider"]
    )
    heatmaps, gaze_feat = backbone.run(None, {"image": batch[:, None, :, :]})

    head = GazeHead.load(head_path)
    landmarks = softargmax2d(heatmaps)
    gaze = head(gaze_feat, landmarks)

    ref_landmarks = reference[1]
    ref_gaze = reference[2]

    print(f"sample    : {sample}")
    print(f"heatmaps  : max abs diff {np.abs(heatmaps - reference[0][:, -1]).max():.3e}")
    print(f"landmarks : max abs diff {np.abs(landmarks - ref_landmarks).max():.3e}")
    print(f"gaze      : max abs diff {np.abs(gaze - ref_gaze).max():.3e}")
    print(f"  reference gaze (rad): {ref_gaze.ravel()}")
    print(f"  split     gaze (rad): {gaze.ravel()}")


def _read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(path).resize((160, 96)).convert("L"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split the EyeGaze ONNX into an NPU friendly backbone")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--backbone", type=str, default=str(DEFAULT_BACKBONE))
    parser.add_argument("--head", type=str, default=str(DEFAULT_HEAD))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    split(Path(args.model), Path(args.backbone), Path(args.head))
    if args.verify:
        verify(Path(args.model), Path(args.backbone), Path(args.head))


if __name__ == "__main__":
    main()
