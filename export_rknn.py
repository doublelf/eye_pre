"""Compile the EyeGaze backbone for the RK3576 NPU with rknn-toolkit2.

Rockchip publishes aarch64 wheels of the toolkit that contain the full
compiler (``librknnc.so``), so the ONNX -> RKNN conversion runs **on this
board**.

    python export_rknn.py                      # convert + report
    python export_rknn.py --quantization       # int8 model (much less accurate)

The script has to run with the conversion virtual environment, which holds
the toolkit (and its older numpy/onnx pins)::

    .venv-rknn/bin/python export_rknn.py

The resulting ``assets/eyegaze_rk3576.rknn`` is used by the runtime::

    python video_infer.py --backend rknn
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ASSETS = HERE / "assets"
DEFAULT_BACKBONE = ASSETS / "eyegaze_backbone.onnx"
DEFAULT_CALIB = ASSETS / "calib"
DEFAULT_RKNN = ASSETS / "eyegaze_rk3576.rknn"

MODEL_NAME = "eyegaze"
TARGET_PLATFORM = "rk3576"


def prepare_calibration(
    calib_dir: Path = DEFAULT_CALIB,
    png_dir: Path | None = None,
    limit: int = 100,
) -> Path:
    """Make sure the calibration set is in a format the toolkit accepts.

    The toolkit reads ``.npy`` files whose layout matches the model input
    (1, 96, 160) - the format produced by the model surgery step.
    PNG copies are written as a fallback for toolkits that only accept
    images.
    """
    npy_files = sorted(calib_dir.glob("*.npy"))
    if not npy_files:
        raise SystemExit(
            f"no calibration samples in {calib_dir}\n"
            "Create them with:  python export_rknn.py --prepare-only"
        )

    png_dir = png_dir or (calib_dir.parent / "calib_png")
    png_dir.mkdir(parents=True, exist_ok=True)
    for old in png_dir.glob("*.png"):
        old.unlink()

    import cv2

    for path in npy_files[:limit]:
        image = np.load(path)
        image = image.reshape(96, 160)
        cv2.imwrite(str(png_dir / f"{path.stem}.png"), image)

    print(f"calibration: {len(npy_files)} samples in {calib_dir} "
          f"({min(limit, len(npy_files))} png copies in {png_dir})")
    return calib_dir


def convert(
    backbone: Path = DEFAULT_BACKBONE,
    rknn_path: Path = DEFAULT_RKNN,
    calib_dir: Path = DEFAULT_CALIB,
    quantization: bool = False,
    optimization_level: int = 3,
    batch_size: int = 2,
) -> Path:
    """Convert the backbone ONNX into an RKNN model.

    The default is a **float16** model (no int8 quantisation).  The
    EyeGaze heatmaps are tiny logits (~1e-2) which a softargmax with
    beta=100 amplifies, so int8 activation quantisation destroys them
    (10-20 deg of gaze error).  float16 keeps the error below 0.1 deg.
    """
    # the shim must be imported before rknn.api
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import rknn_shim  # noqa: F401
    from rknn.api import RKNN

    if not backbone.exists():
        raise SystemExit(
            f"backbone not found: {backbone}\n"
            "Create it with:  python -m split_model"
        )

    # the toolkit drops its optimisation passes (check*.onnx) into the
    # working directory -> run the conversion in a scratch folder
    rknn = RKNN(verbose=True)
    work_dir = rknn_path.parent / "rknn_build"
    work_dir.mkdir(parents=True, exist_ok=True)
    cwd = Path.cwd()
    os.chdir(work_dir)
    try:
        _convert(rknn, backbone, rknn_path, calib_dir, quantization,
                 optimization_level, batch_size)
    finally:
        os.chdir(cwd)
        for leftover in work_dir.glob("check*.onnx"):
            leftover.unlink()

    print(f"\nRKNN model: {rknn_path} "
          f"({rknn_path.stat().st_size / 1e6:.1f} MB)")
    return rknn_path


def _convert(rknn, backbone: Path, rknn_path: Path, calib_dir: Path,
             quantization: bool, optimization_level: int, batch_size: int) -> None:
    print(f"toolkit : rknn-toolkit2 (target {TARGET_PLATFORM})")
    print(f"model   : {backbone}")
    print(f"dataset : {calib_dir}")
    print(f"mode    : {'int8 quantised' if quantization else 'float16 (recommended)'}\n")

    config = dict(
        target_platform=TARGET_PLATFORM,
        quantized_algorithm="normal",
        optimization_level=optimization_level,
        # the runtime normalises the raw uint8 image with (x - mean) / std
        mean_values=[0],
        std_values=[255],
    )
    if quantization:
        config["quantized_dtype"] = "w8a8"
    else:
        config["float_dtype"] = "float16"
    ret = rknn.config(**config)
    if ret != 0:
        raise SystemExit(f"rknn.config failed with code {ret}")

    # the batch dimension is dynamic in the exported graph, which the
    # toolkit does not accept -> pin it
    ret = rknn.load_onnx(
        str(backbone),
        inputs=["image"],
        input_size_list=[[1, 1, 96, 160]],
    )
    if ret != 0:
        raise SystemExit(f"rknn.load_onnx failed with code {ret}")

    ret = rknn.build(
        do_quantization=quantization,
        dataset=str(calib_dir) if quantization else None,
        rknn_batch_size=batch_size,
    )
    if ret != 0:
        raise SystemExit(f"rknn.build failed with code {ret}")

    ret = rknn.export_rknn(str(rknn_path))
    if ret != 0:
        raise SystemExit(f"rknn.export_rknn failed with code {ret}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export EyeGaze for the RK3576 NPU")
    parser.add_argument("--backbone", type=str, default=str(DEFAULT_BACKBONE))
    parser.add_argument("--rknn", type=str, default=str(DEFAULT_RKNN))
    parser.add_argument("--calib", type=str, default=str(DEFAULT_CALIB))
    parser.add_argument("--quantization", action="store_true",
                        help="build an int8 model (faster, but ~10-20 deg of gaze error)")
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="NPU batch size (both eyes in one call)")
    args = parser.parse_args()

    try:
        import rknn  # noqa: F401
    except ImportError:
        raise SystemExit(
            "rknn-toolkit2 is not available in this interpreter.\n"
            "Run the conversion with the dedicated environment:\n"
            "    .venv-rknn/bin/python export_rknn.py"
        )

    prepare_calibration(Path(args.calib))
    convert(
        Path(args.backbone),
        Path(args.rknn),
        Path(args.calib),
        quantization=args.quantization,
        optimization_level=args.optimization_level,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
