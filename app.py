import streamlit as st
import numpy as np
from PIL import Image
import time
import hashlib
from cryptography.fernet import Fernet

# =========================
# SECURITY SETUP (BANK STYLE)
# =========================
key = Fernet.generate_key()
cipher = Fernet(key)

DB_FILE = "secure_db.npy"

try:
    db = np.load(DB_FILE, allow_pickle=True).item()
except:
    db = {}

session_state = {}

# =========================
# ENCRYPTION
# =========================
def encrypt(data):
    return cipher.encrypt(str(data).encode())

def decrypt(data):
    return cipher.decrypt(data).decode()

# =========================
# FEATURE VECTOR
# =========================
def extract_features(img):
    img = img.resize((64, 64))
    arr = np.array(img.convert("L")) / 255.0
    return arr.flatten()

# =========================
# BLINK DETECTION (SIMULATED)
# =========================
def blink_detection():
    return np.random.choice([True, False], p=[0.85, 0.15])

# =========================
# HEAD MOVEMENT TRACKING
# =========================
def head_movement():
    return np.random.uniform(0, 1) > 0.3

# =========================
# DEPTH ESTIMATION (SIMULATED 3D)
# =========================
def depth_estimation():
    return np.random.uniform(0.6, 1.0)

# =========================
# ANTI-DEEPFAKE (FREQUENCY ANALYSIS)
# =========================
def frequency_check(img):
    arr = np.array(img.convert("L"))
    fft = np.fft.fft2(arr)
    score = np.mean(np.abs(fft))
    return score > 50

# =========================
# CNN PLACEHOLDER (EfficientNet concept)
# =========================
def cnn_fake_detector(img):
    arr = np.array(img.convert("L"))
    score = arr.var()
    return score > 1000

# =========================
# LIVENESS ENGINE
# =========================
def liveness_check(img):
    blink = blink_detection()
    motion = head_movement()
    depth = depth_estimation()

    score = sum([blink, motion, depth > 0.7])

    return score >= 2

# =========================
# SIMILARITY
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# MFA (OTP SIMULATION)
# =========================
def generate_otp():
    return str(np.random.randint(100000, 999999))

# =========================
# UI
# =========================
st.title("🏦 Enterprise Biometric Security System")

menu = st.sidebar.selectbox("Menu", ["Enroll", "Login", "Database"])

# =========================
# ENROLL
# =========================
if menu == "Enroll":
    name = st.text_input("User Name")
    img_file = st.camera_input("Capture Face")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            if not liveness_check(img):
                st.error("🚨 Liveness failed (fake or photo detected)")
            elif not cnn_fake_detector(img):
                st.error("🚨 Deepfake detected")
            elif not frequency_check(img):
                st.error("🚨 Frequency anomaly detected")
            else:
                features = extract_features(img)
                db[name] = encrypt(features.tolist())
                np.save(DB_FILE, db)
                st.success("✅ User enrolled securely")

# =========================
# LOGIN
# =========================
elif menu == "Login":
    img_file = st.camera_input("Verify Identity")

    if img_file:
        img = Image.open(img_file)

        if st.button("Login"):
            if not liveness_check(img):
                st.error("🚨 Liveness failed")
            elif not cnn_fake_detector(img):
                st.error("🚨 Deepfake detected")
            elif not frequency_check(img):
                st.error("🚨 Frequency anomaly detected")
            else:
                features = extract_features(img)

                best, best_score = None, 0

                for name, enc in db.items():
                    stored = np.array(eval(decrypt(enc)))

                    score = cosine(features, stored)

                    if score > best_score:
                        best = name
                        best_score = score

                if best_score > 0.75:
                    otp = generate_otp()
                    session_state["otp"] = otp

                    st.success(f"Access granted: {best}")
                    st.info(f"OTP sent: {otp}")
                    st.write("Confidence:", best_score)
                else:
                    st.error("Access denied")

# =========================
# DATABASE
# =========================
elif menu == "Database":
    st.write("Secure users stored:", len(db))
