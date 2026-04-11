import streamlit as st
import numpy as np
import mediapipe as mp
from PIL import Image
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Biometric System", layout="centered")

st.title("🔐 Face Recognition Biometric System")

# =========================
# DATABASE
# =========================
DB_FILE = "database.npy"

if os.path.exists(DB_FILE):
    database = np.load(DB_FILE, allow_pickle=True).item()
else:
    database = {}

def save_db():
    np.save(DB_FILE, database)

# =========================
# FEATURE EXTRACTION (NO CV2)
# =========================
def get_feature_vector(img):
    img = img.resize((64, 64))
    img = np.array(img.convert("L"))  # grayscale
    return img.flatten() / 255.0

# =========================
# ANTI-SPOOFING (simple blur)
# =========================
def is_real_face(img):
    img_np = np.array(img.convert("L"))
    blur = img_np.var()
    return blur > 40

# =========================
# COSINE SIMILARITY
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Enroll User", "Login", "Database"])

# =========================
# ENROLL USER
# =========================
if menu == "Enroll User":
    st.subheader("👤 Register User")

    name = st.text_input("Enter name")
    img_file = st.camera_input("Take photo")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            if not is_real_face(img):
                st.error("❌ Fake or blurry image detected!")
            else:
                database[name] = get_feature_vector(img)
                save_db()
                st.success(f"✅ User {name} enrolled!")

# =========================
# LOGIN
# =========================
elif menu == "Login":
    st.subheader("🔐 Login")

    img_file = st.camera_input("Take login photo")

    if img_file:
        img = Image.open(img_file)

        if st.button("Login"):
            if not is_real_face(img):
                st.error("❌ Attack detected (blur/fake image)")
            else:
                input_vec = get_feature_vector(img)

                best_match = None
                best_score = 0

                for name, vec in database.items():
                    score = cosine_similarity(input_vec, vec)

                    if score > best_score:
                        best_score = score
                        best_match = name

                if best_score > 0.75:
                    st.success(f"✅ Access Granted: {best_match}")
                else:
                    st.error("❌ Access Denied")

                st.write("Confidence:", round(best_score, 2))

# =========================
# DATABASE VIEW
# =========================
elif menu == "Database":
    st.subheader("📊 Registered Users")

    if len(database) == 0:
        st.warning("No users yet")
    else:
        for k in database.keys():
            st.write("👤", k)
