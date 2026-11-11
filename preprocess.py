"""Torch-free preprocessing and visualisation helpers.

These functions are shared by the inference path and the CLI tools, so
they live in a module without any heavyweight dependency.
"""

from __future__ import annotations

import cv2
import numpy as np

IMAGE_W = 160
IMAGE_H = 96
IRIS_CENTER = np.array([80.0, 48.0], dtype=np.float32)  # center of 160x96


def preprocess_eye_crop(img: np.ndarray, side: str = "left") -> np.ndarray:
    """Grayscale eye crop -> normalized 96x160 model input."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (IMAGE_W, IMAGE_H))
    img = cv2.equalizeHist(img)
    img = img.astype(np.float32) / 255.0
    if side == "right":
        img = np.fliplr(img).copy()
    return img


def draw_gaze(
    image_in: np.ndarray,
    eye_pos: np.ndarray,
    pitchyaw: np.ndarray,
    length: float | None = None,
    thickness: int = 2,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """Draw the gaze direction arrow on the image."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)

    H, W = image_out.shape[:2]
    if length is None:
        length = 0.35 * float(min(H, W))

    dx = -np.sin(float(pitchyaw[1]))
    dy = np.sin(float(pitchyaw[0]))

    norm = np.hypot(dx, dy)
    if norm > 1e-6:
        dx /= norm
        dy /= norm

    dx *= length
    dy *= length

    start_pt = np.round(eye_pos).astype(np.int32)
    end_pt = np.array([eye_pos[0] + dx, eye_pos[1] + dy], dtype=np.float32)

    end_pt[0] = np.clip(end_pt[0], 0, W - 1)
    end_pt[1] = np.clip(end_pt[1], 0, H - 1)
    end_pt_int = tuple(np.round(end_pt).astype(np.int32))

    cv2.arrowedLine(
        image_out,
        tuple(start_pt),
        end_pt_int,
        color,
        thickness,
        cv2.LINE_AA,
        tipLength=0.2,
    )
    return image_out
