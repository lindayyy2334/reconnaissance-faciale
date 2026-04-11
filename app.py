import streamlit as st
import numpy as np
from PIL import Image
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Biometric System", layout="centered")
st.title("🔐 Secure Biometric System (Fixed)")

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
# LIVENESS (FIXED & CALIBRATED)
# =========================
def liveness_check(img):
    arr = np.array(img.convert("L"))

    variance = arr.var()
    edges = np.mean(np.abs(np.gradient(arr)))

    score = variance + edges * 30

    # ✅ CALIBRATED THRESHOLDS
    if score < 800:
        return False, score
    elif score < 1100:
        return False, score
    else:
        return True, score

# =========================
# ANTI SCREEN / PHOTO ATTACK
# =========================
def screen_attack(img):
    arr = np.array(img.convert("L"))

    fft = np.fft.fft2(arr)
    noise = np.mean(np.abs(fft))

    # écran = trop structuré
    return noise < 90

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
            ok, score = liveness_check(img)

            st.write("Liveness score:", round(score, 2))

            if not ok:
                st.error("❌ Fake / low quality image")
            elif not screen_attack(img):
                st.error("❌ Screen / phone attack detected")
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
            ok, score = liveness_check(img)

            st.write("Liveness score:", round(score, 2))

            if not ok:
                st.error("❌ Liveness failed")
            elif not screen_attack(img):
                st.error("❌ Screen / photo attack detected")
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

                st.write("Match confidence:", round(best_score, 2))

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
