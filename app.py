from flask import Flask, request, jsonify
import cv2
import numpy as np
import requests
import tempfile
import os
import face_recognition

app = Flask(__name__)

def download_image(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(response.content)
    tmp.close()
    return tmp.name

@app.route("/verify", methods=["POST"])
def verify():
    data           = request.get_json()
    live_url       = data.get("liveUrl")
    registered_url = data.get("registeredUrl")

    if not live_url or not registered_url:
        return jsonify({"error": "liveUrl and registeredUrl are required"}), 400

    live_path = registered_path = None
    try:
        live_path       = download_image(live_url)
        registered_path = download_image(registered_url)

        live_img       = face_recognition.load_image_file(live_path)
        registered_img = face_recognition.load_image_file(registered_path)

        live_encodings       = face_recognition.face_encodings(live_img)
        registered_encodings = face_recognition.face_encodings(registered_img)

        if not live_encodings:
            return jsonify({"match": False, "reason": "no_face_in_live_image"})
        if not registered_encodings:
            return jsonify({"match": False, "reason": "no_face_in_registered_image"})

        match = face_recognition.compare_faces(
            [registered_encodings[0]],
            live_encodings[0],
            tolerance=0.5
        )[0]

        return jsonify({"match": bool(match)})

    except Exception as e:
        return jsonify({"match": False, "reason": str(e)}), 500

    finally:
        for path in [live_path, registered_path]:
            if path and os.path.exists(path):
                os.remove(path)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
