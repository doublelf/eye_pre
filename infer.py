"""Single-image EyeGaze inference on the RK3576 NPU.

Two modes:
  * default      : the input is already a cropped eye (as in the AI Hub demo asset)
  * --auto-crop  : the input is a full photo/frame; MediaPipe locates both eyes,
                   each eye is cropped and evaluated, and the two results are averaged.

Examples
--------
    python infer.py                                  # bundled sample eye crop
    python infer.py --image my_eye.png --side right
    python infer.py --image portrait.jpg --auto-crop --output output/portrait_gaze.png
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from preprocess import IRIS_CENTER, draw_gaze, preprocess_eye_crop
from engine import EYE_LEFT, EYE_RIGHT, GazeEngine

DEFAULT_IMAGE = Path(__file__).parent / "assets" / "input_image.png"


def auto_crop_mode(args: argparse.Namespace) -> None:
    """Locate both eyes in a full image, evaluate them and annotate the result."""
    import cv2

    from face_roi import FaceEyeProvider, crop_eye

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    provider = FaceEyeProvider(expand=args.expand)
    eyes = provider.detect(frame, 0)
    if eyes is None:
        raise SystemExit("No face found in the image")
    provider.close()

    engine = GazeEngine(args.rknn, args.head)
    left_roi = crop_eye(frame, eyes.left_box)
    right_roi = crop_eye(frame, eyes.right_box)
    gaze = engine.predict([left_roi, right_roi], [EYE_LEFT, EYE_RIGHT])

    pitch = float(np.mean(gaze[:, 0]))
    yaw = float(np.mean(gaze[:, 1]))
    diff = abs(math.degrees(gaze[0][1] - gaze[1][1]))

    print(f"eye ROI      : left {eyes.left_box}  right {eyes.right_box}")
    print(f"left  eye    : pitch {math.degrees(gaze[0][0]):+.2f} deg  yaw {math.degrees(gaze[0][1]):+.2f} deg")
    print(f"right eye    : pitch {math.degrees(gaze[1][0]):+.2f} deg  yaw {math.degrees(gaze[1][1]):+.2f} deg")
    print(f"averaged     : pitch {math.degrees(pitch):+.2f} deg  yaw {math.degrees(yaw):+.2f} deg")
    print(f"eye agreement: {diff:.2f} deg (|yaw_left - yaw_right|)")

    # Draw gaze arrows on the full image (both eyes use the same camera-frame formula).
    h, w = frame.shape[:2]
    arrow_len = 0.6 * (eyes.left_box[2] - eyes.left_box[0])
    for origin, g, box, color in (
        (eyes.left_iris, gaze[0], eyes.left_box, (0, 255, 0)),
        (eyes.right_iris, gaze[1], eyes.right_box, (0, 200, 255)),
    ):
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        dx = -math.sin(float(g[1]))
        dy = math.sin(float(g[0]))
        end = (int(origin[0] + dx * arrow_len), int(origin[1] + dy * arrow_len))
        cv2.arrowedLine(frame, (int(origin[0]), int(origin[1])), end, (0, 0, 255), 3,
                        cv2.LINE_AA, tipLength=0.25)
    cv2.putText(frame, f"pitch {math.degrees(pitch):+.2f} deg  yaw {math.degrees(yaw):+.2f} deg",
                (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    out_path = Path(args.output) if args.output else Path("output") / "gaze_image.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), frame)
    print(f"Saved annotated image: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="EyeGaze single-image inference")
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="eye crop image path")
    parser.add_argument("--side", type=str, default="left", choices=["left", "right"])
    parser.add_argument("--output", type=str, default=None, help="path for annotated output image")
    parser.add_argument("--auto-crop", action="store_true",
                        help="input is a full image; locate both eyes with MediaPipe")
    parser.add_argument("--rknn", type=str, default=str(Path(__file__).parent / "assets" / "eyegaze_rk3576.rknn"),
                        help="RK3576 NPU model with the EyeGaze backbone")
    parser.add_argument("--head", type=str, default=str(Path(__file__).parent / "assets" / "eyegaze_head.npz"),
                        help="host side gaze head weights (softargmax + FC layers)")
    parser.add_argument("--expand", type=float, default=2.2)
    args = parser.parse_args()

    if args.auto_crop:
        auto_crop_mode(args)
        return

    engine = GazeEngine(args.rknn, args.head)
    print(f"Backend: RK3576 NPU ({args.rknn})")

    # cv2.resize (bilinear) is used on purpose: it is the same resampling
    # the ROI crops go through in face_roi.crop_eye, so the single image
    # path and the video path see the same pixels (PIL's bicubic resize
    # differed by up to 1 degree of pitch).
    image_np = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if image_np is None:
        raise SystemExit(f"Cannot read image: {args.image}")
    image_np = cv2.resize(image_np, (160, 96))

    gaze = engine.predict([image_np], [args.side])[0]
    pitch, yaw = float(gaze[0]), float(gaze[1])

    print(f"gaze pitch: {pitch:.4f} rad ({math.degrees(pitch):.2f} deg)")
    print(f"gaze yaw  : {yaw:.4f} rad ({math.degrees(yaw):.2f} deg)")
    print(f"gaze angle: {math.degrees(math.atan2(yaw, pitch)):.2f} deg (atan2(yaw, pitch))")

    proc = preprocess_eye_crop(image_np, args.side)
    vis = cv2.cvtColor((proc * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    annotated = draw_gaze(vis, IRIS_CENTER, gaze)
    out_path = Path(args.output) if args.output else Path("output") / "gaze_result.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(annotated).save(out_path)
    print(f"Saved annotated image: {out_path}")


if __name__ == "__main__":
    main()
