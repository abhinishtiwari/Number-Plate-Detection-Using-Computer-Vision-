#!/usr/bin/env python
"""Command line interface for the Number Plate AI pipeline.

This wraps the same backend package the API uses, so the CLI and the web app can
never disagree. The previous root-level CLI drove a second, parallel
implementation in src/ whose OCR fallback returned the literal string
"DETECTED_PLATE" with a fixed 0.70 confidence.

Examples
--------
    python main.py
    python main.py --image samples/demo_plate.jpg --output output/demo_plate.jpg
    python main.py --video clip.mp4 --csv output/detections.csv
    python main.py --image samples/demo_plate.jpg --json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2

from backend import __version__
from backend.config import VIDEO_FRAME_STRIDE

logger = logging.getLogger("cli")


def process_frame(image):
    """Load the shared pipeline lazily so CLI imports do not mount the web app."""
    from backend.main import process_frame as shared_process_frame

    return shared_process_frame(image)


def annotate(image, plates):
    """Draw each box with the text that was read from that same box."""
    output = image.copy()
    for plate in plates:
        x, y, w, h = plate["box"]
        readable = bool(plate["text"])
        colour = (102, 255, 0) if readable else (32, 176, 255)
        label = plate["text"] or "unreadable"
        if plate["confidence"] is not None:
            label = f"{label} ({plate['confidence']:.1f}%)"

        cv2.rectangle(output, (x, y), (x + w, y + h), colour, 3)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        tag_y = y - th - 10 if y - th - 10 >= 0 else y
        cv2.rectangle(output, (x, tag_y), (x + tw + 10, tag_y + th + 10), colour, -1)
        cv2.putText(output, label, (x + 5, tag_y + th + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return output


def describe(plate: dict) -> str:
    parts = [plate["text"] or "unreadable"]
    if plate["confidence"] is not None:
        parts.append(f"{plate['confidence']:.1f}%")
    location = ", ".join(p for p in (plate["state_name"], plate["city"]) if p)
    if location:
        parts.append(location)
    if plate["full_rto_code"]:
        parts.append(plate["full_rto_code"])
    return " | ".join(parts)


def log_to_csv(path: Path, plates: list[dict], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["timestamp_utc", "source", "plate", "valid_format",
                             "confidence", "ocr_engine", "state", "rto_code", "city"])
        for plate in plates:
            writer.writerow([
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                source,
                plate["text"] or "",
                plate["is_valid_format"],
                "" if plate["confidence"] is None else f"{plate['confidence']:.1f}",
                plate["ocr_engine"],
                plate["state_name"] or "",
                plate["full_rto_code"] or "",
                plate["city"] or "",
            ])


def run_image(args) -> int:
    path = Path(args.image)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    image = cv2.imread(str(path))
    if image is None:
        print(f"error: could not decode {path}", file=sys.stderr)
        return 2

    plates = process_frame(image)
    if args.json:
        print(json.dumps({"source": str(path), "plates": plates}, indent=2))
    elif not plates:
        print("No number plate found.")
    else:
        print(f"Found {len(plates)} plate region(s):")
        for i, plate in enumerate(plates, 1):
            print(f"  {i}. {describe(plate)}")

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out), annotate(image, plates)):
            print(f"error: could not write annotated image to {out}", file=sys.stderr)
            return 1
        print(f"Annotated image written to {out}", file=sys.stderr if args.json else sys.stdout)
    if args.csv:
        log_to_csv(Path(args.csv), plates, str(path))
    return 0


def run_video(args) -> int:
    source = int(args.video) if args.video.isdigit() else args.video
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        print(f"error: could not open video source {args.video!r}", file=sys.stderr)
        return 2

    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if size[0] <= 0 or size[1] <= 0:
            print("error: video reports invalid frame dimensions", file=sys.stderr)
            capture.release()
            return 2
        writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        if not writer.isOpened():
            print(f"error: could not create output video {args.output!r}", file=sys.stderr)
            capture.release()
            return 2

    seen: dict[str, dict] = {}
    frame_index = 0
    progress = sys.stderr if args.json else sys.stdout
    print("Processing video. Press 'q' in the preview window to stop.", file=progress)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame_plates: list[dict] = []
            if frame_index % args.stride == 0:
                frame_plates = process_frame(frame)
                for plate in frame_plates:
                    if plate["text"] and plate["text"] not in seen:
                        seen[plate["text"]] = plate
                        print(f"  frame {frame_index}: {describe(plate)}", file=progress)

            annotated = annotate(frame, frame_plates)
            if writer:
                writer.write(annotated)
            if args.show:
                cv2.imshow("Number Plate AI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
    finally:
        capture.release()
        if writer:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print(f"Done. {len(seen)} unique plate(s) across {frame_index} frame(s).", file=progress)
    if args.json:
        print(json.dumps({"source": args.video, "plates": list(seen.values())}, indent=2))
    if args.csv:
        log_to_csv(Path(args.csv), list(seen.values()), args.video)
    return 0


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Number Plate AI - run the local dashboard or process media from the command line."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", help="path to an image file")
    source.add_argument("--video", help="path to a video file, or a webcam index such as 0")
    parser.add_argument("--output", help="where to write the annotated image/video")
    parser.add_argument("--csv", help="append detections to this CSV log")
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument("--stride", type=positive_int, default=VIDEO_FRAME_STRIDE,
                        help="process every Nth video frame (default: %(default)s)")
    parser.add_argument("--show", action="store_true", help="show a live preview window (video only)")
    parser.add_argument("--host", default="127.0.0.1", help="dashboard bind address (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="dashboard port (default: %(default)s)")
    parser.add_argument("--reload", action="store_true", help="reload automatically while developing")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def run_server(args) -> int:
    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Number Plate AI {__version__} starting at {url}")
    print("Keep this window open. Press Ctrl+C to stop.")
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.getLogger().setLevel(logging.DEBUG if args.verbose else logging.INFO)
    if args.image:
        return run_image(args)
    if args.video:
        return run_video(args)
    if args.output or args.csv or args.json or args.show:
        print("error: media options require --image or --video", file=sys.stderr)
        return 2
    return run_server(args)


if __name__ == "__main__":
    raise SystemExit(main())
