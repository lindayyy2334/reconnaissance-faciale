import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Secure Biometrics", layout="centered")

st.title("🛡️ Face Authentication + Liveness Detection")

DB_FILE = "db.npy"

if os.path.exists(DB_FILE):
    db = np.load(DB_FILE, allow_pickle=True).item()
else:
    db = {}

def save_db():
    np.save(DB_FILE, db)

# =========================
# FEATURE EXTRACTION
# =========================
def extract_features(img):
    img = img.resize((64, 64))
    img = np.array(img.convert("L")) / 255.0
    return img.flatten()

# =========================
# ANTI-FAKE (liveness)
# =========================
def liveness_score(img):
    arr = np.array(img.convert("L"))

    # 1. blur / texture
    variance = arr.var()

    # 2. edge richness (fake images are smoother)
    edges = np.mean(np.abs(np.gradient(arr)))

    score = variance + edges * 50
    return score

def is_live(img):
    score = liveness_score(img)
    return score > 1200, score  # threshold adjustable

# =========================
# SIMILARITY
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# UI
# =========================
menu = st.sidebar.selectbox("Menu", ["Enroll", "Login", "Database"])

# =========================
# ENROLL
# =========================
if menu == "Enroll":
    name = st.text_input("Name")
    img_file = st.camera_input("Take face photo")

    if img_file and name:
        img = Image.open(img_file)

        live, score = is_live(img)

        st.write("Liveness score:", score)

        if st.button("Enroll"):
            if not live:
                st.error("❌ Fake image detected (no liveness)")
            else:
                db[name] = extract_features(img)
                save_db()
                st.success("✅ User enrolled")

# =========================
# LOGIN
# =========================
elif menu == "Login":
    img_file = st.camera_input("Take login photo")

    if img_file:
        img = Image.open(img_file)

        live, score = is_live(img)
        st.write("Liveness score:", score)

        if st.button("Login"):
            if not live:
                st.error("🚨 Fake / photo attack detected")
            else:
                vec = extract_features(img)

                best, best_score = None, 0

                for name, v in db.items():
                    s = cosine(vec, v)
                    if s > best_score:
                        best, best_score = name, s

                if best_score > 0.75:
                    st.success(f"Access granted: {best}")
                else:
                    st.error("Access denied")

                st.write("Similarity:", best_score)

# =========================
# DATABASE
# =========================
elif menu == "Database":
    st.write(db)
