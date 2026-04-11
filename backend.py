from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

# =========================
# LIVENESS (blink + motion simple)
# =========================
def detect_motion(frames):
    diffs = []
    for i in range(len(frames)-1):
        diff = np.mean(np.abs(frames[i] - frames[i+1]))
        diffs.append(diff)

    return np.mean(diffs) > 5  # mouvement réel

# =========================
# FAKE FACE EMBEDDING (placeholder FaceNet)
# =========================
def get_embedding(face):
    face = cv2.resize(face, (160,160))
    return face.flatten() / 255.0

# =========================
# API TEST
# =========================
@app.get("/")
def home():
    return {"status": "AI Biometric System Running"}

# =========================
# FACE VERIFY
# =========================
@app.post("/verify/")
async def verify(file: UploadFile = File(...)):
    image = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(image, cv2.IMREAD_COLOR)

    embedding = get_embedding(img)

    return {
        "status": "processed",
        "embedding_size": len(embedding)
    }
