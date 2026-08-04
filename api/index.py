"""
Minimal Flask API for Vercel's Python runtime.

WARNING (read this before you spend time debugging):
This will very likely fail to deploy on Vercel due to platform limits:
  - fast-plate-ocr + open-image-models + onnxruntime + opencv easily
    exceed Vercel's 250MB (standard) function bundle size limit.
  - Even if the build succeeds, loading ONNX models fresh on every
    "cold start" (which happens often on serverless) will be slow and
    may exceed the execution time limit.
  - Model weight files need to download on first use, which requires
    writable disk - serverless functions only get limited /tmp space.

This file exists so you can see the *actual* error Vercel throws,
rather than guessing. Once you hit the wall, the recommended fix is
to move to Hugging Face Spaces / Render / Railway instead.
"""
import os
os.environ["HOME"] = "/tmp"
os.environ["XDG_CACHE_HOME"] = "/tmp/.cache"
import base64
import io
from flask import Flask, request, jsonify
import cv2
import numpy as np

from fast_alpr import ALPR

app = Flask(__name__)

# NOTE: this loads on every cold start - this alone can time out on Vercel
alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v2-global-model",
)


@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "FastALPR API is running"})


@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    result = alpr.draw_predictions(img)

    plates = []
    for r in result.results:
        if r.ocr and r.ocr.text:
            plates.append({
                "text": r.ocr.text,
                "region": r.ocr.region,
                "confidence": r.detection.confidence,
            })

    # Encode annotated image back to base64 to return in JSON
    _, buffer = cv2.imencode(".png", result.image)
    annotated_b64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "plates": plates,
        "annotated_image_base64": annotated_b64,
    })


# Vercel's Python runtime looks for a WSGI-compatible "app" object
