from flask import Flask, request, jsonify
import numpy as np
import requests
import tempfile
import os
import insightface
from insightface.app import FaceAnalysis

app = Flask(__name__)

# تحميل النموذج مرة وحدة عند بدء السيرفر
face_app = FaceAnalysis(name="buffalo_sc", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(320, 320))

def download_image(url: str) -> np.ndarray:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(response.content)
    tmp.close()
    import cv2
    img = cv2.imread(tmp.name)
    os.remove(tmp.name)
    return img

def get_embedding(img: np.ndarray):
    faces = face_app.get(img)
    if not faces:
        return None
    return faces[0].embedding

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

@app.route("/verify", methods=["POST"])
def verify():
    data           = request.get_json()
    live_url       = data.get("liveUrl")
    registered_url = data.get("registeredUrl")

    if not live_url or not registered_url:
        return jsonify({"error": "liveUrl and registeredUrl are required"}), 400

    try:
        live_img       = download_image(live_url)
        registered_img = download_image(registered_url)

        live_emb       = get_embedding(live_img)
        registered_emb = get_embedding(registered_img)

        if live_emb is None:
            return jsonify({"match": False, "reason": "no_face_in_live_image"})
        if registered_emb is None:
            return jsonify({"match": False, "reason": "no_face_in_registered_image"})

        similarity = cosine_similarity(live_emb, registered_emb)
        match      = similarity >= 0.4

        return jsonify({"match": bool(match), "similarity": round(similarity, 4)})

    except Exception as e:
        return jsonify({"match": False, "reason": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
