"""EyeGaze inference engine - RK3576 NPU backend.

The whole model chain lives on the Rockchip NPU:

    eye crop (160x96 gray)
        -> preprocess (resize, equalizeHist, /255, flip for the right eye)
        -> .rknn backbone : heatmaps (34, 48, 80) + gaze_feat (7680)
        -> host : softargmax -> landmarks (34, 2)
        -> host : FC1 -> ReLU -> FC2 -> (pitch, yaw)

The softargmax decoder and the fully connected gaze head stay on the host
because neither can be expressed on the NPU (see :mod:`gaze_head`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent

EYE_LEFT = "left"  # subject's left eye
EYE_RIGHT = "right"  # subject's right eye

DEFAULT_RKNN = HERE / "assets" / "eyegaze_rk3576.rknn"
DEFAULT_HEAD = HERE / "assets" / "eyegaze_head.npz"


class GazeEngine:
    """Runs the EyeGaze model on the Rockchip NPU.

    Side handling follows the reference implementation: the right eye is
    mirrored on input and its yaw sign is inverted, so both eyes' outputs
    are expressed in the same camera-frame convention.
    """

    def __init__(
        self,
        rknn_path: str | Path = DEFAULT_RKNN,
        head_path: str | Path = DEFAULT_HEAD,
    ) -> None:
        from rknn_backend.gaze import RknnGazeEngine

        self.rknn_path = Path(rknn_path)
        self.head_path = Path(head_path)
        self.engine = RknnGazeEngine(self.rknn_path, self.head_path)

    def predict(self, eyes: list[np.ndarray], sides: list[str]) -> np.ndarray:
        """Predict gaze for a batch of 160x96 gray eye crops.

        Returns an (N, 2) array of [pitch, yaw] in radians.
        """
        return self.engine.predict(eyes, sides)

    def close(self) -> None:
        self.engine.close()
