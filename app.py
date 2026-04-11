import streamlit as st
import numpy as np
from PIL import Image
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Biometric Security System", layout="centered")
st.title("🔐 Secure Biometric System (Anti-Spoofing)")

DB_FILE = "db.npy"

# =========================
# DATABASE
# =========================
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
    arr = np.array(img.convert("L")) / 255.0
    return arr.flatten()

# =========================
# ANTI-FAKE LIVENESS (AMÉLIORÉ)
# =========================
def liveness_check(img):
    arr = np.array(img.convert("L"))

    variance = arr.var()
    edges = np.mean(np.abs(np.gradient(arr)))

    # score combiné
    score = variance + edges * 50

    return score > 1600

# =========================
# ANTI PHONE / SCREEN ATTACK
# =========================
def screen_attack_detection(img):
    arr = np.array(img.convert("L"))

    # écran = patterns trop réguliers
    freq = np.abs(np.fft.fft2(arr))
    noise_level = np.mean(freq)

    return noise_level < 80  # écran = souvent très structuré

# =========================
# ANTI-DEEPFAKE (LIGHT)
# =========================
def deepfake_check(img):
    arr = np.array(img.convert("L"))
    fft = np.fft.fft2(arr)
    score = np.mean(np.abs(fft))

    return 30 < score < 120

# =========================
# SIMILARITY
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Enroll", "Login", "Database"])

# =========================
# ENROLL USER
# =========================
if menu == "Enroll":
    name = st.text_input("Enter Name")
    img_file = st.camera_input("Capture Face")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            if not liveness_check(img):
                st.error("❌ Fake detected (no liveness)")
            elif not screen_attack_detection(img):
                st.error("❌ Screen / phone attack detected")
            elif not deepfake_check(img):
                st.error("❌ Deepfake suspected")
            else:
                db[name] = extract_features(img)
                save_db()
                st.success(f"✅ User {name} enrolled")

# =========================
# LOGIN
# =========================
elif menu == "Login":
    img_file = st.camera_input("Verify Identity")

    if img_file:
        img = Image.open(img_file)

        if st.button("Login"):
            if not liveness_check(img):
                st.error("❌ Liveness failed")
            elif not screen_attack_detection(img):
                st.error("❌ Phone / photo attack detected")
            elif not deepfake_check(img):
                st.error("❌ Deepfake detected")
            else:
                features = extract_features(img)

                best_user = None
                best_score = 0

                for name, vec in db.items():
                    score = cosine(features, vec)
                    if score > best_score:
                        best_user = name
                        best_score = score

                if best_score > 0.75:
                    st.success(f"✅ Access Granted: {best_user}")
                else:
                    st.error("❌ Access Denied")

                st.write("Confidence:", round(best_score, 2))

# =========================
# DATABASE
# =========================
elif menu == "Database":
    st.subheader("📊 Registered Users")

    if len(db) == 0:
        st.warning("No users registered")
    else:
        for k in db.keys():
            st.write("👤", k)
