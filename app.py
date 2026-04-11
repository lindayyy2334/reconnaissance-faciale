import streamlit as st
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="Biometric System", layout="centered")
st.title("🔐 Simple Biometric System (Cloud Safe)")

DB_FILE = "database.npy"

if os.path.exists(DB_FILE):
    database = np.load(DB_FILE, allow_pickle=True).item()
else:
    database = {}

def save_db():
    np.save(DB_FILE, database)

def get_vector(img):
    img = img.resize((64, 64))
    img = np.array(img.convert("L"))
    return img.flatten() / 255.0

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

menu = st.sidebar.selectbox("Menu", ["Enroll", "Login", "Database"])

# ENROLL
if menu == "Enroll":
    name = st.text_input("Name")
    img_file = st.camera_input("Take photo")

    if img_file and name:
        img = Image.open(img_file)

        if st.button("Enroll"):
            database[name] = get_vector(img)
            save_db()
            st.success("User enrolled!")

# LOGIN
elif menu == "Login":
    img_file = st.camera_input("Take photo")

    if img_file:
        img = Image.open(img_file)

        if st.button("Login"):
            vec = get_vector(img)

            best, score = None, 0

            for name, v in database.items():
                s = cosine(vec, v)
                if s > score:
                    best, score = name, s

            if score > 0.75:
                st.success(f"Access granted: {best}")
            else:
                st.error("Access denied")

            st.write("Confidence:", round(score, 2))

# DATABASE
elif menu == "Database":
    st.write(database)
