"""Eye region extraction from video frames using MediaPipe FaceLandmarker.

The EyeGaze model consumes a *cropped eye* image (96x160 grayscale), so a video
pipeline needs to locate the eyes first. MediaPipe FaceLandmarker gives 478
facial landmarks (including iris centres), which makes the eye ROI accurate and
stable across frames.

Landmark conventions (canonical 468-point face mesh):
    subject right eye : outer 33, inner 133, iris centre 468
    subject left  eye : inner 362, outer 263, iris centre 473
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions

DEFAULT_FACE_MODEL = Path(__file__).parent / "assets" / "face_landmarker.task"

# Eye contour rings (eyelid outline) for a tighter ROI.
RIGHT_EYE_RING = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE_RING = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

RIGHT_IRIS_CENTER = 468
LEFT_IRIS_CENTER = 473

# Inner / outer eye corners (used for the gaze offset sanity check).
RIGHT_EYE_INNER = 133
RIGHT_EYE_OUTER = 33
LEFT_EYE_INNER = 362
LEFT_EYE_OUTER = 263

# Face oval edges and nose tip (head rotation proxies).
FACE_OVAL_RIGHT = 234
FACE_OVAL_LEFT = 454
NOSE_TIP = 1

# Model input aspect (w x h) = 160 x 96.
MODEL_ASPECT = 160 / 96


@dataclass
class EyeRegion:
    """Eye ROI (in frame pixels) for both eyes."""

    left_box: tuple[int, int, int, int]  # subject's left eye  (x0, y0, x1, y1)
    right_box: tuple[int, int, int, int]  # subject's right eye (x0, y0, x1, y1)
    left_iris: tuple[float, float]  # subject's left iris centre in frame pixels
    right_iris: tuple[float, float]  # subject's right iris centre in frame pixels
    left_corners: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
    )  # subject's left eye (inner, outer) corners
    right_corners: tuple[tuple[float, float], tuple[float, float]] = (
        (0.0, 0.0),
        (0.0, 0.0),
    )  # subject's right eye (inner, outer) corners
    head_yaw_deg: float = 0.0  # proxy for head rotation about the vertical axis
    head_roll_deg: float = 0.0  # eye line tilt in the frame

    def left_offset_x(self) -> float:
        """Iris offset from the eye-corner midpoint (camera frame, +x = image right)."""
        mid = (self.left_corners[0][0] + self.left_corners[1][0]) / 2
        return self.left_iris[0] - mid

    def right_offset_x(self) -> float:
        mid = (self.right_corners[0][0] + self.right_corners[1][0]) / 2
        return self.right_iris[0] - mid


def _expand_box(
    pts: np.ndarray,
    center: tuple[float, float],
    expand: float,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Grow the eye bbox around the iris centre, keeping the model aspect ratio."""
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    box_w = max(float(x1 - x0), 1.0)
    box_h = max(float(y1 - y0), 1.0)

    width = box_w * expand
    height = width / MODEL_ASPECT
    if height < box_h * expand:
        height = box_h * expand
        width = height * MODEL_ASPECT

    cx, cy = center
    half_w = width / 2
    half_h = height / 2

    fh, fw = frame_shape[:2]
    nx0 = int(round(max(0.0, min(cx - half_w, fw - 1))))
    ny0 = int(round(max(0.0, min(cy - half_h, fh - 1))))
    nx1 = int(round(min(float(fw), max(cx + half_w, nx0 + 1))))
    ny1 = int(round(min(float(fh), max(cy + half_h, ny0 + 1))))
    return nx0, ny0, nx1, ny1


class FaceEyeProvider:
    """Locate both eyes in every frame of a video."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_FACE_MODEL,
        num_faces: int = 1,
        expand: float = 2.2,
    ) -> None:
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=num_faces,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self.expand = expand
        self._last_timestamp = -1

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> EyeRegion | None:
        """Return eye ROIs for a frame, or None when no face is found."""
        rgb = cv2_rgb(frame_bgr)
        timestamp = max(int(timestamp_ms), self._last_timestamp + 1)
        self._last_timestamp = timestamp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp)

        if not result.face_landmarks:
            return None

        lm = result.face_landmarks[0]
        h, w = frame_bgr.shape[:2]
        to_px = np.array([[p.x * w, p.y * h] for p in lm], dtype=np.float32)

        right_pts = to_px[RIGHT_EYE_RING]
        left_pts = to_px[LEFT_EYE_RING]
        right_iris = tuple(to_px[RIGHT_IRIS_CENTER])
        left_iris = tuple(to_px[LEFT_IRIS_CENTER])

        # Head rotation proxies: nose offset between the two face-oval edges,
        # and the tilt of the eye line.
        face_right = to_px[FACE_OVAL_RIGHT]
        face_left = to_px[FACE_OVAL_LEFT]
        nose = to_px[NOSE_TIP]
        half_width = max(abs(float(face_left[0] - face_right[0])) / 2.0, 1.0)
        nose_offset = float(nose[0]) - (float(face_right[0]) + float(face_left[0])) / 2.0
        head_yaw = math.degrees(math.asin(max(-1.0, min(1.0, nose_offset / half_width))))
        head_roll = eye_line_angle(
            (tuple(to_px[LEFT_EYE_INNER]), tuple(to_px[LEFT_EYE_OUTER]))
        )

        return EyeRegion(
            left_box=_expand_box(left_pts, left_iris, self.expand, frame_bgr.shape),
            right_box=_expand_box(right_pts, right_iris, self.expand, frame_bgr.shape),
            left_iris=left_iris,
            right_iris=right_iris,
            left_corners=(tuple(to_px[LEFT_EYE_INNER]), tuple(to_px[LEFT_EYE_OUTER])),
            right_corners=(tuple(to_px[RIGHT_EYE_INNER]), tuple(to_px[RIGHT_EYE_OUTER])),
            head_yaw_deg=head_yaw,
            head_roll_deg=head_roll,
        )

    def close(self) -> None:
        self._landmarker.close()


def cv2_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR (OpenCV) -> RGB (MediaPipe)."""
    return np.ascontiguousarray(frame_bgr[:, :, ::-1])


def eye_line_angle(corners: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Angle (degrees) of the eye line in the frame, normalized to (-90, 90].

    The eye line is a line, not a vector, so the 180 degree ambiguity between
    the two eyes is removed by the normalization.
    """
    (x0, y0), (x1, y1) = corners
    angle = math.degrees(math.atan2(y1 - y0, x1 - x0))
    while angle > 90.0:
        angle -= 180.0
    while angle <= -90.0:
        angle += 180.0
    return angle


def crop_eye(
    frame_bgr: np.ndarray,
    box: tuple[int, int, int, int],
    iris: tuple[float, float] | None = None,
    corners: tuple[tuple[float, float], tuple[float, float]] | None = None,
    deskew: bool = True,
) -> np.ndarray:
    """Crop an eye ROI and convert it to the model's 160x96 grayscale input.

    With `deskew`, the crop is rotated so the eye line is horizontal. Head roll
    otherwise makes the model read the rotation as a yaw offset (and the two
    eyes are rolled in opposite directions, so the two estimates drift apart).
    """
    import cv2

    x0, y0, x1, y1 = box

    if deskew and iris is not None and corners is not None:
        angle = eye_line_angle(corners)
        scale = max((x1 - x0) / 160.0, 1e-6)  # source px per output px
        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        # dst -> src: the output x-axis follows the eye line.
        m = np.array(
            [
                [cos_t * scale, -sin_t * scale],
                [sin_t * scale, cos_t * scale],
            ]
        )
        offset = np.array([iris[0], iris[1]]) - m @ np.array([80.0, 48.0])
        transform = np.hstack([m, offset.reshape(2, 1)])
        patch = cv2.warpAffine(
            frame_bgr,
            transform,
            (160, 96),
            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
    else:
        patch = frame_bgr[y0:y1, x0:x1]

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 96))
