"""
Run FastALPR license plate detection on a video file.

- Reads an input video frame by frame
- Detects + OCRs license plates on each frame
- Draws boxes and plate text on each frame
- Writes an annotated video to disk
- Shows a live preview window while processing (press 'q' to stop early)
- Prints each newly-seen plate number once
- Prints a summary of all unique plates at the end

Usage:
    python run_video.py
    python run_video.py --input traffic.mp4 --output output_video.mp4
    python run_video.py --input traffic.mp4 --no-preview
    python run_video.py --input 0        (use webcam instead of a file)
"""

import argparse
import sys

import cv2

from fast_alpr import ALPR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastALPR on a video.")
    parser.add_argument(
        "--input",
        "-i",
        default="traffic.mp4",
        help="Path to input video file, or '0' for webcam. Default: traffic.mp4",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output_video.mp4",
        help="Path to save the annotated output video. Default: output_video.mp4",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window (useful for headless/servers).",
    )
    parser.add_argument(
        "--detector-model",
        default="yolo-v9-t-384-license-plate-end2end",
        help="Detector model name from open-image-models hub.",
    )
    parser.add_argument(
        "--ocr-model",
        default="cct-xs-v2-global-model",
        help="OCR model name from fast-plate-ocr hub.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Allow "--input 0" to mean webcam index 0
    input_source: str | int = args.input
    if isinstance(input_source, str) and input_source.isdigit():
        input_source = int(input_source)

    print("Loading models (first run downloads them, may take a moment)...")
    alpr = ALPR(
        detector_model=args.detector_model,
        ocr_model=args.ocr_model,
    )

    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"ERROR: Could not open video source: {args.input}")
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"ERROR: Could not open output video for writing: {args.output}")
        cap.release()
        return 1

    show_preview = not args.no_preview
    frame_num = 0
    seen_plates: set[str] = set()

    print(f"Processing '{args.input}' -> '{args.output}'")
    if show_preview:
        print("Press 'q' in the preview window to stop early.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            result = alpr.draw_predictions(frame)
            writer.write(result.image)

            for r in result.results:
                if r.ocr and r.ocr.text and r.ocr.text not in seen_plates:
                    seen_plates.add(r.ocr.text)
                    print(f"Frame {frame_num}: New plate detected -> {r.ocr.text}")

            if show_preview:
                cv2.imshow("FastALPR - press 'q' to quit", result.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    break

            if total_frames > 0 and frame_num % 30 == 0:
                print(f"Processed {frame_num}/{total_frames} frames...")
            elif total_frames <= 0 and frame_num % 30 == 0:
                print(f"Processed {frame_num} frames...")

    finally:
        cap.release()
        writer.release()
        if show_preview:
            cv2.destroyAllWindows()

    print(f"\nDone. Annotated video saved to: {args.output}")
    if seen_plates:
        print(f"All unique plates seen ({len(seen_plates)}):")
        for plate in sorted(seen_plates):
            print(f"  - {plate}")
    else:
        print("No plates were detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
