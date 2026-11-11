"""Validate the RKNN pipeline against the reference ONNX model.

Checks, on real eye crops:

1. the RKNN backbone on the NPU against the full exported model - the
   error comes from the float16 arithmetic of the accelerator,
2. optionally the int8 model (which is much less accurate, see
   ``export_rknn.py``).

Usage
-----
    python -m rknn_backend.verify
    python -m rknn_backend.verify --rknn assets/eyegaze_rk3576.rknn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ASSETS = ROOT / "assets"

DEFAULT_MODEL = ASSETS / "model_src" / "eyegaze.onnx"
DEFAULT_RKNN = ASSETS / "eyegaze_rk3576.rknn"
DEFAULT_HEAD = ASSETS / "eyegaze_head.npz"
DEFAULT_CALIB = ASSETS / "calib"


def load_samples(calib_dir: Path, limit: int) -> np.ndarray:
    """Load the calibration crops (already 160x96 gray, 0..255)."""
    files = sorted(calib_dir.glob("*.npy"))
    if not files:
        raise SystemExit(
            f"no calibration samples in {calib_dir}\n"
            "Create them with:  python export_rknn.py --prepare-only"
        )
    samples = []
    for path in files[:limit]:
        data = np.load(path).astype(np.float32) / 255.0
        samples.append(data.reshape(96, 160))
    return np.stack(samples)


def reference_gaze(model_path: Path, samples: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    name = session.get_inputs()[0].name
    # the exported reference model takes a 3D image (N, 96, 160)
    batch = samples.astype(np.float32)
    return session.run(None, {name: batch})[2]


def rknn_gaze(rknn_path: Path, head: Path, samples: np.ndarray) -> np.ndarray:
    from gaze_head import GazeHead, softargmax2d
    from rknn_backend.gaze import RknnGazeEngine

    engine = RknnGazeEngine(rknn_path, head)
    try:
        heatmaps, gaze_feat = engine.infer_backbone(samples)
    finally:
        engine.close()

    landmarks = softargmax2d(heatmaps)
    return GazeHead.load(head)(gaze_feat, landmarks)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the RKNN eye gaze pipeline")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--rknn", type=str, default=str(DEFAULT_RKNN))
    parser.add_argument("--head", type=str, default=str(DEFAULT_HEAD))
    parser.add_argument("--calib", type=str, default=str(DEFAULT_CALIB))
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="max allowed |difference| in degrees")
    args = parser.parse_args()

    rknn_path = Path(args.rknn)
    if not rknn_path.exists():
        raise SystemExit(
            f"RKNN model not found: {rknn_path}\n"
            "Build it with:  .venv-rknn/bin/python export_rknn.py"
        )

    # This check compares against the reference ONNX model, so onnxruntime is
    # an optional development dependency - skip instead of failing when it is
    # not installed.
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        print("[SKIP] onnxruntime is not installed - "
              "cannot compare against the reference model")
        print("       install it with:  pip install onnxruntime")
        raise SystemExit(0)

    samples = load_samples(Path(args.calib), args.samples)
    print(f"samples: {len(samples)} eye crops from {args.calib}\n")

    reference = reference_gaze(Path(args.model), samples)
    print(f"reference pitch {np.degrees(reference[:, 0]).mean():+.3f} deg   "
          f"yaw {np.degrees(reference[:, 1]).mean():+.3f} deg\n")

    result = rknn_gaze(rknn_path, Path(args.head), samples)
    diff = np.degrees(np.abs(result - reference))
    print(f"RKNN (NPU) vs ONNX reference")
    print(f"  max |dPitch| {diff[:, 0].max():.4f} deg   "
          f"max |dYaw| {diff[:, 1].max():.4f} deg   "
          f"mean |d| {diff.mean():.4f} deg")

    if diff.max() > args.tolerance:
        print(f"\nFAIL: the deviation exceeds {args.tolerance} deg")
        raise SystemExit(1)
    print(f"\nOK: the NPU matches the reference within {args.tolerance} deg")


if __name__ == "__main__":
    main()
