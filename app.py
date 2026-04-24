import streamlit as st
import cv2
import numpy as np
import face_recognition
import hashlib, hmac, os
from PIL import Image

st.set_page_config(page_title="Lab Biométrie", page_icon="🔐", layout="wide")

# ─── Auth optionnelle ───────────────────────────────────────────
if "password" in st.secrets:
    pwd = st.sidebar.text_input("Mot de passe", type="password")
    if pwd != st.secrets["password"]:
        st.warning("Accès protégé — entrez le mot de passe")
        st.stop()

# ─── Session state ───────────────────────────────────────────────
if "database" not in st.session_state:
    st.session_state.database = {}

database = st.session_state.database

# ─── Fonctions cœur ──────────────────────────────────────────────
def enroll_user(name, image_array):
    rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if encodings:
        database[name] = {"template": encodings[0].tolist()}
        return True
    return False

def recognize_user(image_array):
    rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return None, 0.0
    best_match, best_score = None, 0.0
    for name, data in database.items():
        dist = face_recognition.face_distance(
            [np.array(data["template"])], encodings[0])[0]
        score = 1 - dist
        if score > best_score:
            best_match, best_score = name, score
    return best_match, round(best_score, 4)

def anti_spoofing(image_array):
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture = np.std(gray.astype(np.float32))
    hsv = cv2.cvtColor(image_array, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1].mean()
    scores = {"Netteté": (blur, blur > 50),
              "Texture": (texture, texture > 20),
              "Saturation": (sat, sat > 30)}
    return scores

def bio_hash(encoding, salt=None):
    if salt is None:
        salt = os.urandom(32)
    rng = np.random.default_rng(int.from_bytes(salt[:8], "big"))
    mat = rng.standard_normal((len(encoding), len(encoding) // 2))
    proj = encoding @ mat
    vec = (proj > 0).astype(int)
    h = hmac.new(salt, vec.tobytes(), hashlib.sha256).hexdigest()
    return h, salt

# ─── Navigation ──────────────────────────────────────────────────
st.sidebar.title("🔐 Lab Biométrie")
page = st.sidebar.radio("Navigation", [
    "📸 Enrollment",
    "🔓 Login",
    "🛡️ Protection avancée",
    "📊 Comparatif"
])

# ─── PAGE 1 : Enrollment ─────────────────────────────────────────
if page == "📸 Enrollment":
    st.title("📸 Enregistrement d'un utilisateur")
    name = st.text_input("Nom de l'utilisateur")
    img_file = st.camera_input("Prenez une photo")

    if img_file and name:
        img = np.array(Image.open(img_file))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if st.button("✅ Enregistrer"):
            if enroll_user(name, img_bgr):
                st.success(f"✅ {name} enregistré avec succès !")
            else:
                st.error("❌ Aucun visage détecté")

    if database:
        st.info(f"👥 Utilisateurs enregistrés : {', '.join(database.keys())}")

# ─── PAGE 2 : Login ──────────────────────────────────────────────
elif page == "🔓 Login":
    st.title("🔓 Connexion biométrique")
    img_file = st.camera_input("Capturez votre visage")

    if img_file:
        img = np.array(Image.open(img_file))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        st.subheader("🔍 Anti-Spoofing")
        scores = anti_spoofing(img_bgr)
        cols = st.columns(3)
        all_pass = True
        for i, (label, (val, ok)) in enumerate(scores.items()):
            with cols[i]:
                color = "🟢" if ok else "🔴"
                st.metric(f"{color} {label}", f"{val:.1f}")
            if not ok:
                all_pass = False

        if not all_pass:
            st.error("🚨 Attaque possible détectée !")
        else:
            st.success("✅ Vivacité confirmée")
            name, score = recognize_user(img_bgr)
            st.subheader("🧠 Résultat")
            if score > 0.5:
                st.success(f"✅ Accès accordé — **{name}** (confiance : {score:.2%})")
                st.balloons()
            else:
                st.error(f"❌ Accès refusé (meilleur score : {score:.2%})")

# ─── PAGE 3 : Protection avancée ─────────────────────────────────
elif page == "🛡️ Protection avancée":
    st.title("🛡️ Méthodes de protection")
    img_file = st.camera_input("Image de démonstration")

    if img_file:
        img = np.array(Image.open(img_file))
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)

        if encodings:
            enc = encodings[0]
            tab1, tab2, tab3 = st.tabs(["Bio-Hashing", "Cancelable", "Anonyme"])

            with tab1:
                st.markdown("**Principe** : Le template est transformé en hash via un sel secret. Impossible de retrouver le visage original.")
                h, salt = bio_hash(enc)
                st.code(f"Hash : {h[:32]}...\nSel  : {salt.hex()[:32]}...")
                st.success("✅ Template brut NON stocké")

            with tab2:
                st.markdown("**Principe** : Permutation + transformation non-linéaire. Si compromis → changer la clé → nouveau template.")
                key = os.urandom(16)
                rng = np.random.default_rng(int.from_bytes(key[:8], "big"))
                perm = rng.permutation(len(enc))
                transformed = np.sign(enc[perm] - rng.uniform(-0.1, 0.1, len(enc)))
                st.code(f"Clé : {key.hex()[:16]}...\nTemplate transformé (128D) : [{', '.join([str(int(x)) for x in transformed[:8]])}...]")

            with tab3:
                st.markdown("**Principe** : Aucun nom stocké. Identifiant anonyme via SHA-256.")
                salt = os.urandom(16)
                anon_id = hashlib.sha256(enc.tobytes() + salt).hexdigest()
                st.code(f"ID anonyme : {anon_id[:32]}...\nNom stocké : AUCUN")
        else:
            st.warning("Aucun visage détecté")

# ─── PAGE 4 : Comparatif ─────────────────────────────────────────
elif page == "📊 Comparatif":
    st.title("📊 Comparatif des méthodes")
    import pandas as pd
    df = pd.DataFrame([
        {"Méthode": "Template brut", "Révocable": "❌", "Non-inversible": "❌", "Anonyme": "❌", "Complexité": "⭐"},
        {"Méthode": "Bio-Hashing",   "Révocable": "✅", "Non-inversible": "✅", "Anonyme": "⚠️", "Complexité": "⭐⭐"},
        {"Méthode": "Cancelable",    "Révocable": "✅", "Non-inversible": "✅", "Anonyme": "⚠️", "Complexité": "⭐⭐"},
        {"Méthode": "Anonyme",       "Révocable": "⚠️", "Non-inversible": "✅", "Anonyme": "✅", "Complexité": "⭐⭐⭐"},
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.info("👥 Utilisateurs en session : " + str(len(database)))
