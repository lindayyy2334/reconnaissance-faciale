import streamlit as st
import numpy as np
from PIL import Image
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Secure Biometrics", layout="centered")
st.title("🔐 Biometric Security System")

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
# LIVENESS (ANTI PHOTO)
# =========================
def liveness(img):
    arr = np.array(img.convert("L"))
    return arr.var() > 900

# =========================
# ANTI-DEEPFAKE (simple FFT)
# =========================
def deepfake_check(img):
    arr = np.array(img.convert("L"))
    fft = np.fft.fft2(arr)
    score = np.mean(np.abs(fft))
    return score > 40

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
# ENROLL
# =========================
if menu == "Enroll":
    name = st.text_input("Enter Name")
    img_file = st.camera_input("Take Photo")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            if not liveness(img):
                st.error("❌ Fake image detected")
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
            if not liveness(img):
                st.error("❌ Liveness failed")
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
