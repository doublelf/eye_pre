"""RKNN (Rockchip NPU) backend for the EyeGaze model.

The RK3576 NPU runs the convolutional backbone of EyeNet, while the
softargmax landmark decoder and the two fully connected gaze layers stay
on the host - see :mod:`gaze_head`.

    eye crop (160x96 gray)
        -> preprocess (resize, equalizeHist, /255, flip for the right eye)
        -> .rknn backbone : heatmaps (34, 48, 80) + gaze_feat (7680)
        -> host : softargmax -> landmarks (34, 2)
        -> host : FC1 -> ReLU -> FC2 -> (pitch, yaw)

The .rknn model is produced by ``export_rknn.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gaze_head import GazeHead, softargmax2d
from preprocess import preprocess_eye_crop

EYE_LEFT = "left"
EYE_RIGHT = "right"

HEATMAP_SHAPE = (34, 48, 80)
FEATURE_SIZE = 7680


def _output_index(names: list[str], wanted: str) -> int:
    for i, name in enumerate(names):
        if name.split("/")[-1] == wanted or name == wanted:
            return i
    raise KeyError(f"output {wanted!r} not found (outputs: {names})")


class RknnGazeEngine:
    """Runs the EyeGaze backbone on the Rockchip NPU."""

    def __init__(
        self,
        rknn_path: str | Path,
        head_path: str | Path,
        channels: int = 1,
        core_mask: int | None = None,
    ) -> None:
        from rknn_backend.runtime import RknnModel

        self.rknn_path = Path(rknn_path)
        self.head_path = Path(head_path)

        if not self.rknn_path.exists():
            raise FileNotFoundError(
                f"RKNN model not found: {self.rknn_path}\n"
                "Build it with:  python export_rknn.py"
            )
        if not self.head_path.exists():
            raise FileNotFoundError(
                f"gaze head weights not found: {self.head_path}\n"
                "Create them with:  python -m split_model"
            )

        self.model = RknnModel(self.rknn_path, core_mask=core_mask)
        self.head = GazeHead.load(self.head_path)
        self.channels = channels

    def infer_backbone(
        self, processed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run the compiled backbone on already preprocessed images.

        Parameters
        ----------
        processed
            Batch of float32 images in [0, 1] with shape (N, 96, 160).
        """
        processed = np.asarray(processed, dtype=np.float32)
        count = processed.shape[0]

        # The model bakes the /255 scaling into its input preprocessing
        # (mean=0, std=255), so the raw 8 bit image is what the NPU wants.
        quantised = np.rint(processed * 255.0).astype(np.uint8)
        nhwc = quantised[:, :, :, None]
        if self.channels == 3:
            nhwc = np.repeat(nhwc, 3, axis=2)

        # one call for both eyes: the NPU call overhead dominates, a batch
        # of two costs the same as a batch of one
        outputs = self.model.infer(nhwc)
        heatmaps = outputs[0].reshape(count, *HEATMAP_SHAPE)
        gaze_feat = outputs[1].reshape(count, FEATURE_SIZE)
        return heatmaps, gaze_feat

    def predict(self, eyes: list[np.ndarray], sides: list[str]) -> np.ndarray:
        """Predict gaze for a batch of 160x96 gray eye crops.

        Returns an (N, 2) array of [pitch, yaw] in radians.
        """
        if not eyes:
            return np.zeros((0, 2), dtype=np.float32)

        batch = np.stack(
            [preprocess_eye_crop(eye, side) for eye, side in zip(eyes, sides)]
        ).astype(np.float32)
        heatmaps, gaze_feat = self.infer_backbone(batch)

        landmarks = softargmax2d(heatmaps)
        gaze = self.head(gaze_feat, landmarks)

        for i, side in enumerate(sides):
            if side == EYE_RIGHT:
                gaze[i, 1] = -gaze[i, 1]
        return gaze.astype(np.float32)

    def close(self) -> None:
        self.model.close()
