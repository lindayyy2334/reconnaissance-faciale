import streamlit as st
import requests
from PIL import Image

st.title("🔐 Biometric AI System")

file = st.camera_input("Capture face")

if file:
    if st.button("Verify"):
        response = requests.post(
            "http://localhost:8000/verify/",
            files={"file": file.getvalue()}
        )

        st.json(response.json())
