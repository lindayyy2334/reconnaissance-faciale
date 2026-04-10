import streamlit as st
import face_recognition
import numpy as np
import cv2
import os

# ======================
# DATABASE
# ======================
if os.path.exists("database.npy"):
    database = np.load("database.npy", allow_pickle=True).item()
else:
    database = {}

# ======================
# SAVE DATABASE
# ======================
def save_db():
    np.save("database.npy", database)

# ======================
# ENROLL FACE
# ======================
def enroll(name, image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) > 0:
        database[name] = encodings[0]
        save_db()
        return "User enrolled successfully"
    return "No face detected"

# ======================
# LOGIN
# ======================
def login(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)

    if len(encodings) == 0:
        return "No face detected", 0

    input_encoding = encodings[0]

    best_match = None
    best_score = 0

    for name, enc in database.items():
        distance = face_recognition.face_distance([enc], input_encoding)[0]
        score = 1 - distance

        if score > best_score:
            best_score = score
            best_match = name

    if best_score > 0.5:
        return f"Access granted: {best_match}", best_score
    else:
        return "Access denied", best_score

# ======================
# STREAMLIT UI
# ======================
st.title("🔐 Face Recognition Login System")

menu = st.sidebar.selectbox("Menu", ["Enroll User", "Login"])

# ======================
# ENROLL
# ======================
if menu == "Enroll User":
    st.subheader("Register New User")

    name = st.text_input("Enter name")

    img_file = st.camera_input("Take a photo")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if st.button("Enroll"):
            msg = enroll(name, img)
            st.success(msg)

# ======================
# LOGIN
# ======================
if menu == "Login":
    st.subheader("Face Login")

    img_file = st.camera_input("Take login photo")

    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)

        if st.button("Login"):
            result, score = login(img)

            st.write(result)
            st.write("Confidence:", round(score, 2))
