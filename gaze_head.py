"""CPU post-processing head for the EyeGaze model.

The NPU cannot express the softargmax landmark decoder (softmax over
48x80 spatial positions followed by two weighted sums) nor the 7748-input
fully connected gaze head, so both stay on the host:

    heatmaps (34, 48, 80)  --softargmax-->  landmarks (34, 2)
    gaze_feat (7680)       ------------------>  concat --> FC1 --> FC2

The math here is a bit-exact numpy port of ``util/softargmax.py`` and of
the ``gaze_fc1`` / ``gaze_fc2`` layers of ``models/eyenet.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def softargmax2d(heatmaps: np.ndarray, beta: float = 100.0) -> np.ndarray:
    """Decode heatmaps into landmark coordinates.

    Parameters
    ----------
    heatmaps
        Array of shape (N, L, H, W).
    beta
        Softmax sharpness (100.0 in the reference implementation).

    Returns
    -------
    Array of shape (N, L, 2) holding (row, col) coordinates scaled to
    (H - 1) / (W - 1), i.e. the same values the original model produces.
    """
    heatmaps = np.asarray(heatmaps, dtype=np.float32)
    n, landmarks, height, width = heatmaps.shape

    flat = heatmaps.reshape(n * landmarks, height * width) * np.float32(beta)
    flat -= flat.max(axis=-1, keepdims=True)
    np.exp(flat, out=flat)
    flat /= flat.sum(axis=-1, keepdims=True)

    # (row, col) coordinates of every spatial position
    rows = np.linspace(0.0, 1.0, height, dtype=np.float32)
    cols = np.linspace(0.0, 1.0, width, dtype=np.float32)
    grid = np.empty((height * width, 2), dtype=np.float32)
    grid[:, 0] = np.repeat(rows, width)
    grid[:, 1] = np.tile(cols, height)

    coords = flat @ grid  # (n * landmarks, 2)
    coords[:, 0] *= height - 1
    coords[:, 1] *= width - 1
    return coords.reshape(n, landmarks, 2)


@dataclass
class GazeHead:
    """The two fully connected layers that turn features into gaze angles."""

    fc1_weight: np.ndarray  # (256, in_features)
    fc1_bias: np.ndarray  # (256,)
    fc2_weight: np.ndarray  # (2, 256)
    fc2_bias: np.ndarray  # (2,)

    @classmethod
    def load(cls, path: str | Path) -> "GazeHead":
        data = np.load(str(path))
        return cls(
            fc1_weight=data["fc1_weight"].astype(np.float32),
            fc1_bias=data["fc1_bias"].astype(np.float32),
            fc2_weight=data["fc2_weight"].astype(np.float32),
            fc2_bias=data["fc2_bias"].astype(np.float32),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            fc1_weight=self.fc1_weight,
            fc1_bias=self.fc1_bias,
            fc2_weight=self.fc2_weight,
            fc2_bias=self.fc2_bias,
        )

    def __call__(
        self, gaze_feat: np.ndarray, landmarks: np.ndarray
    ) -> np.ndarray:
        """Return the (N, 2) gaze prediction [pitch, yaw] in radians."""
        gaze_feat = np.asarray(gaze_feat, dtype=np.float32)
        landmarks = np.asarray(landmarks, dtype=np.float32)

        flat_landmarks = landmarks.reshape(landmarks.shape[0], -1)
        features = np.concatenate([gaze_feat, flat_landmarks], axis=1)

        hidden = features @ self.fc1_weight.T + self.fc1_bias
        np.maximum(hidden, 0.0, out=hidden)
        return hidden @ self.fc2_weight.T + self.fc2_bias
