"""Video inference for the EyeGaze model on the RK3576 NPU.

Per-frame pipeline:
    MediaPipe FaceLandmarker -> eye ROI (160x96 gray) -> preprocess
    -> RKNN backbone (NPU) -> softargmax + FC head (host) -> pitch/yaw
    -> overlay gaze arrows on the frame -> write mp4 + per-frame CSV

Usage:
    python video_infer.py
    python video_infer.py --video assets/videos/head-pose-face-detection-female.mp4 --stride 2
    python video_infer.py --show
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import cv2
import numpy as np

from preprocess import draw_gaze, preprocess_eye_crop
from engine import EYE_LEFT, EYE_RIGHT, GazeEngine
from face_roi import FaceEyeProvider, crop_eye

HERE = Path(__file__).parent
# NOTE: assets/head-pose.mp4 is a truncated copy (only 547 of 1615 frames decode),
# so the complete download is used as the default input.
DEFAULT_VIDEO = HERE / "assets" / "videos" / "head-pose-face-detection-female.mp4"
DEFAULT_OUTPUT = HERE / "output" / "gaze_video.mp4"
DEFAULT_CSV = HERE / "output" / "gaze.csv"
DEFAULT_HEAD = HERE / "assets" / "eyegaze_head.npz"
DEFAULT_RKNN = HERE / "assets" / "eyegaze_rk3576.rknn"


def gaze_vector(pitch: float, yaw: float, side: str = EYE_LEFT) -> tuple[float, float]:
    """Gaze direction in frame pixel space (x right, y down).

    Both eyes' outputs are already in the same camera-frame convention
    (the right eye is mirrored on input and its yaw sign is inverted, so
    `yaw` means the same thing for both). The reference implementation draws
    the arrow on the frame with the same formula for either eye
    (see upstream run_with_webcam.py).
    """
    del side  # same convention for both eyes
    dx = -math.sin(yaw)
    dy = math.sin(pitch)
    return dx, dy


def draw_arrow(frame: np.ndarray, origin: tuple[float, float], pitch: float, yaw: float,
               side: str, length: float, color: tuple[int, int, int]) -> None:
    dx, dy = gaze_vector(pitch, yaw, side)
    norm = math.hypot(dx, dy)
    if norm > 1e-6:
        dx, dy = dx / norm, dy / norm
    end = (int(round(origin[0] + dx * length)), int(round(origin[1] + dy * length)))
    cv2.arrowedLine(
        frame,
        (int(round(origin[0])), int(round(origin[1]))),
        end,
        color,
        3,
        cv2.LINE_AA,
        tipLength=0.25,
    )


def paste_roi_thumbnail(frame: np.ndarray, roi_gray: np.ndarray, origin: tuple[int, int],
                        label: str, side: str = EYE_LEFT,
                        pitch: float | None = None, yaw: float | None = None) -> None:
    """Show the model input (160x96 eye crop with gaze arrow) in the corner.

    `origin` is (x, y) in frame pixels.
    """
    proc = preprocess_eye_crop(roi_gray, side)
    vis = cv2.cvtColor((proc * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if pitch is not None and yaw is not None:
        vis = draw_gaze(vis, np.array([80.0, 48.0], dtype=np.float32),
                        np.array([pitch, yaw], dtype=np.float32))
    frame_x, frame_y = origin
    h, w = vis.shape[:2]
    fh, fw = frame.shape[:2]
    if frame_y + h > fh or frame_x + w > fw:
        return
    frame[frame_y : frame_y + h, frame_x : frame_x + w] = vis
    cv2.rectangle(frame, (frame_x, frame_y), (frame_x + w, frame_y + h), (0, 255, 255), 1)
    cv2.putText(frame, label, (frame_x + 4, frame_y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (0, 255, 255), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description="EyeGaze video inference")
    parser.add_argument("--video", type=str, default=str(DEFAULT_VIDEO))
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument("--csv", type=str, default=str(DEFAULT_CSV))
    parser.add_argument("--rknn", type=str, default=str(DEFAULT_RKNN),
                        help="RK3576 NPU model with the EyeGaze backbone")
    parser.add_argument("--head", type=str, default=str(DEFAULT_HEAD),
                        help="host side gaze head weights (softargmax + FC layers)")
    parser.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    parser.add_argument("--expand", type=float, default=2.2, help="eye ROI expansion factor")
    parser.add_argument("--min-eye-width", type=int, default=40,
                        help="min eye ROI width in source px; smaller frames are flagged low-res")
    parser.add_argument("--max-eye-diff", type=float, default=20.0,
                        help="max |yaw_left - yaw_right| in deg; larger frames are flagged unreliable")
    parser.add_argument("--deskew", action="store_true",
                        help="rotate the eye crop to compensate for head roll (off by default, "
                             "matching the upstream implementation)")
    parser.add_argument("--eye", type=str, default="both", choices=["both", "left", "right"])
    parser.add_argument("--show", action="store_true", help="live preview window")
    parser.add_argument("--no-thumbs", action="store_true", help="hide ROI thumbnails")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_path = Path(args.output)
    csv_path = Path(args.csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {video_path}  {width}x{height} @ {fps:.2f} fps  ({total_frames} frames)")
    print(f"Backend: RK3576 NPU | stride={args.stride} | eye={args.eye} | expand={args.expand}")

    provider = FaceEyeProvider(expand=args.expand)
    engine = GazeEngine(args.rknn, args.head)

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    rows: list[dict] = []
    frame_idx = -1
    processed = 0
    detected = 0
    proc_times: list[float] = []
    detect_times: list[float] = []
    infer_times: list[float] = []
    frame_interval = 1000.0 / fps

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx >= args.max_frames:
            break

        run_this = (frame_idx % max(1, args.stride)) == 0
        if not run_this:
            writer.write(frame)
            continue

        t0 = time.perf_counter()
        timestamp = int(frame_idx * frame_interval)
        eyes = provider.detect(frame, timestamp)
        t_detect = time.perf_counter() - t0

        if eyes is None:
            cv2.putText(frame, "no face", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (0, 0, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            rows.append(dict(frame=frame_idx, time_ms=round(timestamp, 1), detected=0,
                             quality="no-face", eye_px=0, eye_diff_deg="",
                             pitch_deg="", yaw_deg="", left_pitch_deg="", left_yaw_deg="",
                             right_pitch_deg="", right_yaw_deg=""))
            processed += 1
            proc_times.append(time.perf_counter() - t0)
            detect_times.append(t_detect)
            continue

        detected += 1
        left_roi = crop_eye(frame, eyes.left_box, eyes.left_iris, eyes.left_corners, args.deskew)
        right_roi = crop_eye(frame, eyes.right_box, eyes.right_iris, eyes.right_corners, args.deskew)

        sides = [EYE_LEFT, EYE_RIGHT]
        crops = [left_roi, right_roi]
        if args.eye == "left":
            sides, crops = [EYE_LEFT], [left_roi]
        elif args.eye == "right":
            sides, crops = [EYE_RIGHT], [right_roi]

        t1 = time.perf_counter()
        gaze = engine.predict(crops, sides)
        infer_times.append(time.perf_counter() - t1)
        pitch = float(np.mean(gaze[:, 0]))
        yaw = float(np.mean(gaze[:, 1]))
        left_pitch = float(gaze[0][0])
        left_yaw = float(gaze[0][1])
        right_pitch = float(gaze[1][0]) if len(gaze) > 1 else float("nan")
        right_yaw = float(gaze[1][1]) if len(gaze) > 1 else float("nan")

        # Eye crop size in source pixels: the model was trained on 80x60 MPIIGaze
        # crops, so tiny ROIs mean the estimate is dominated by upscaling blur.
        eye_px = eyes.left_box[2] - eyes.left_box[0]
        # Self-consistency: the two eyes must agree (they describe the same gaze).
        eye_diff = (
            abs(math.degrees(left_yaw - right_yaw)) if len(gaze) > 1 else float("nan")
        )
        if eye_px < args.min_eye_width:
            quality = "low-res"
        elif not math.isnan(eye_diff) and eye_diff > args.max_eye_diff:
            quality = "eyes-disagree"
        else:
            quality = "ok"
        low_res = quality != "ok"

        # --- overlay ---
        for box, color in ((eyes.left_box, (0, 255, 0)), (eyes.right_box, (0, 200, 255))):
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 1)

        eye_radius = 0.6 * eye_px
        draw_arrow(frame, eyes.left_iris, left_pitch, left_yaw, EYE_LEFT, eye_radius, (0, 0, 255))
        if not math.isnan(right_pitch):
            draw_arrow(frame, eyes.right_iris, right_pitch, right_yaw, EYE_RIGHT, eye_radius, (0, 0, 255))

        cv2.putText(frame, f"pitch {math.degrees(pitch):+6.2f} deg   yaw {math.degrees(yaw):+6.2f} deg",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        info = f"frame {frame_idx}  eye {eye_px}px  eyes-diff {eye_diff:.1f}deg"
        color = (0, 200, 255) if low_res else (255, 255, 255)
        cv2.putText(frame, info, (12, height - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)
        if low_res:
            label = "LOW-RES eye crop" if quality == "low-res" else "EYES DISAGREE"
            cv2.putText(frame, label, (12, 52), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 200, 255), 2, cv2.LINE_AA)

        if not args.no_thumbs:
            paste_roi_thumbnail(frame, left_roi, (width - 170, 10), "L", EYE_LEFT,
                                left_pitch, left_yaw)
            if len(gaze) > 1:
                paste_roi_thumbnail(frame, right_roi, (width - 170, 115), "R", EYE_RIGHT,
                                    right_pitch, right_yaw)

        writer.write(frame)
        rows.append(dict(frame=frame_idx, time_ms=round(timestamp, 1), detected=1,
                         quality=quality, eye_px=eye_px,
                         eye_diff_deg=round(eye_diff, 2) if not math.isnan(eye_diff) else "",
                         pitch_deg=round(math.degrees(pitch), 3),
                         yaw_deg=round(math.degrees(yaw), 3),
                         left_pitch_deg=round(math.degrees(left_pitch), 3),
                         left_yaw_deg=round(math.degrees(left_yaw), 3),
                         right_pitch_deg=round(math.degrees(right_pitch), 3),
                         right_yaw_deg=round(math.degrees(right_yaw), 3)))

        processed += 1
        elapsed = time.perf_counter() - t0
        proc_times.append(elapsed)
        detect_times.append(t_detect)

        if args.show:
            cv2.imshow("eyegaze", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if processed % 50 == 0:
            recent = np.mean(proc_times[-50:])
            print(f"  frame {frame_idx}/{total_frames}  {1.0 / recent:.1f} fps")

    cap.release()
    writer.release()
    provider.close()
    if args.show:
        cv2.destroyAllWindows()

    # --- CSV ---
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["frame"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)

    # --- summary ---
    valid = [r for r in rows if r["detected"] == 1]
    good = [r for r in valid if r["quality"] == "ok"]
    print(f"\nOutput video : {out_path}")
    print(f"Output CSV   : {csv_path}")
    print(f"Frames read  : {frame_idx + 1}  processed: {processed}  face detected: {detected} "
          f"({100 * detected / max(processed, 1):.1f}%)")
    if proc_times:
        print(f"Processing   : {1000 * np.mean(proc_times):.1f} ms/frame "
              f"({1.0 / np.mean(proc_times):.1f} fps), video fps {fps:.1f}")
    if detect_times:
        print(f"  face detect: {1000 * np.mean(detect_times):.1f} ms/frame")
    if infer_times:
        print(f"  gaze infer : {1000 * np.mean(infer_times):.1f} ms/frame (both eyes)")
    if valid:
        eye_px = np.array([r["eye_px"] for r in valid])
        diffs = np.array([r["eye_diff_deg"] for r in valid if r["eye_diff_deg"] != ""])
        print(f"Eye ROI size : {eye_px.mean():.0f} px mean "
              f"(model trained on ~80px crops; < {args.min_eye_width} px = low confidence)")
        if len(diffs):
            print(f"Two-eye check: |yaw_left - yaw_right| mean {diffs.mean():.1f} deg, "
                  f"median {np.median(diffs):.1f} deg  (lower = more reliable)")
    if good:
        p = np.array([r["pitch_deg"] for r in good])
        y = np.array([r["yaw_deg"] for r in good])
        print(f"Pitch (deg)  : mean {p.mean():+.2f}  std {p.std():.2f}  range [{p.min():+.2f}, {p.max():+.2f}]"
              f"   [high-res frames only, n={len(good)}]")
        print(f"Yaw   (deg)  : mean {y.mean():+.2f}  std {y.std():.2f}  range [{y.min():+.2f}, {y.max():+.2f}]")


if __name__ == "__main__":
    main()
