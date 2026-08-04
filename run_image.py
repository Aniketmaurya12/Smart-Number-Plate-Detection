"""
Run FastALPR license plate detection on a single image.

Usage:
    python run_image.py
    python run_image.py --input car.jpg --output output.png
"""

import argparse
import sys

import cv2

from fast_alpr import ALPR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FastALPR on an image.")
    parser.add_argument(
        "--input",
        "-i",
        default="assets/test_image.png",
        help="Path to input image. Default: assets/test_image.png",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="output.png",
        help="Path to save the annotated output image. Default: output.png",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print("Loading models (first run downloads them, may take a moment)...")
    alpr = ALPR(
        detector_model="yolo-v9-t-384-license-plate-end2end",
        ocr_model="cct-xs-v2-global-model",
    )

    result = alpr.draw_predictions(args.input)
    cv2.imwrite(args.output, result.image)

    print(f"\nSaved annotated image to: {args.output}")
    if result.results:
        for r in result.results:
            print(r)
    else:
        print("No plates detected.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
