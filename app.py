from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import tempfile
import os
from deepface import DeepFace

app = Flask(__name__)

def download_image(url: str) -> str:
    """تحميل صورة من URL وحفظها في ملف مؤقت"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    suffix = ".jpg"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(response.content)
    tmp.close()
    return tmp.name

def has_face(image_path: str) -> bool:
    """تحقق إن الصورة فيها وجه بـ OpenCV"""
    img = cv2.imread(image_path)
    if img is None:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

@app.route("/verify", methods=["POST"])
def verify():
    """
    Body (JSON):
    {
      "liveUrl":      "رابط صورة الطالب الحالية من Firebase Storage",
      "registeredUrl":"رابط صورة التسجيل المحفوظة من Firebase Storage"
    }
    Response:
    { "match": true/false }
    """
    data = request.get_json()
    live_url       = data.get("liveUrl")
    registered_url = data.get("registeredUrl")

    if not live_url or not registered_url:
        return jsonify({"error": "liveUrl and registeredUrl are required"}), 400

    live_path       = None
    registered_path = None

    try:
        # تحميل الصورتين
        live_path       = download_image(live_url)
        registered_path = download_image(registered_url)

        # تحقق إن في وجه في الصورة الحالية
        if not has_face(live_path):
            return jsonify({"match": False, "reason": "no_face_detected"})

        # مقارنة الوجهين بـ DeepFace
        result = DeepFace.verify(
            img1_path  = live_path,
            img2_path  = registered_path,
            model_name = "Facenet",
            detector_backend = "opencv",
            enforce_detection = False,
        )
        return jsonify({"match": bool(result["verified"])})

    except Exception as e:
        return jsonify({"match": False, "reason": str(e)}), 500

    finally:
        # حذف الملفات المؤقتة
        for path in [live_path, registered_path]:
            if path and os.path.exists(path):
                os.remove(path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
