import streamlit as st
import numpy as np
import sqlite3
import hashlib
import jwt
import time
from PIL import Image

# =========================
# CONFIG SECURITY
# =========================
SECRET_KEY = "SUPER_SECRET_KEY_PFE"

# =========================
# DATABASE (SQLite)
# =========================
conn = sqlite3.connect("biometric.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS users (
    name TEXT,
    embedding TEXT
)
""")
conn.commit()

# =========================
# FACE EMBEDDING (SIMULATED FaceNet)
# =========================
def extract_embedding(img):
    img = img.resize((112, 112))
    arr = np.array(img.convert("L")) / 255.0
    return arr.flatten()

# =========================
# LIVENESS (ANTI-PHOTO)
# =========================
def liveness_score(img):
    arr = np.array(img.convert("L"))
    return arr.var() > 1000

# =========================
# ANTI-DEEPFAKE (frequency)
# =========================
def frequency_check(img):
    arr = np.array(img.convert("L"))
    fft = np.fft.fft2(arr)
    score = np.mean(np.abs(fft))
    return score > 50

# =========================
# SIMILARITY
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# JWT TOKEN
# =========================
def generate_token(name):
    payload = {"user": name, "time": time.time()}
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

# =========================
# UI
# =========================
st.title("🏦 Enterprise Biometric Security System")

menu = st.sidebar.selectbox("Menu", ["Enroll", "Login", "Users"])

# =========================
# ENROLL
# =========================
if menu == "Enroll":
    name = st.text_input("Name")
    img_file = st.camera_input("Capture Face")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            if not liveness_score(img):
                st.error("❌ Fake detected (no liveness)")
            elif not frequency_check(img):
                st.error("❌ Deepfake suspected")
            else:
                emb = extract_embedding(img)
                c.execute("INSERT INTO users VALUES (?,?)", (name, str(emb.tolist())))
                conn.commit()
                st.success("✅ User enrolled securely")

# =========================
# LOGIN
# =========================
elif menu == "Login":
    img_file = st.camera_input("Verify Identity")

    if img_file:
        img = Image.open(img_file)

        if st.button("Login"):
            if not liveness_score(img):
                st.error("🚨 Liveness failed")
            elif not frequency_check(img):
                st.error("🚨 Deepfake detected")
            else:
                emb = extract_embedding(img)

                c.execute("SELECT * FROM users")
                users = c.fetchall()

                best = None
                best_score = 0

                for name, db_emb in users:
                    db_emb = np.array(eval(db_emb))
                    score = cosine(emb, db_emb)

                    if score > best_score:
                        best = name
                        best_score = score

                if best_score > 0.75:
                    token = generate_token(best)
                    st.success(f"Access granted: {best}")
                    st.write("JWT Token:", token)
                else:
                    st.error("Access denied")

# =========================
# USERS
# =========================
elif menu == "Users":
    c.execute("SELECT name FROM users")
    st.write([u[0] for u in c.fetchall()])
