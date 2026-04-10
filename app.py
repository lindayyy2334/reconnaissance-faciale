
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import os

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Biometric System", layout="centered")

st.title("🔐 Face Recognition Biometric System")

# =========================
# DATABASE (simple local)
# =========================
DB_FILE = "database.npy"

if os.path.exists(DB_FILE):
    database = np.load(DB_FILE, allow_pickle=True).item()
else:
    database = {}

def save_db():
    np.save(DB_FILE, database)

# =========================
# FACE DETECTION (MediaPipe)
# =========================
mp_face = mp.solutions.face_detection
detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.6)

# =========================
# SIMPLE FEATURE EXTRACTION
# (IMPORTANT: lightweight version for cloud)
# =========================
def get_feature_vector(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.flatten() / 255.0

# =========================
# ANTI-SPOOFING (blur check)
# =========================
def is_real_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    return blur > 40

# =========================
# DISTANCE
# =========================
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# =========================
# UI MENU
# =========================
menu = st.sidebar.selectbox("Menu", ["Enroll User", "Login", "Database"])

# =========================
# ENROLL
# =========================
if menu == "Enroll User":
    st.subheader("👤 Register User")

    name = st.text_input("Enter name")

    img_file = st.camera_input("Take photo")

    if img_file and name:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

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
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

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
                    st.write("Confidence:", round(best_score, 2))
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
