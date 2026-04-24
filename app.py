import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import hashlib
import hmac
import os
from PIL import Image

# ─────────────────────────────────────────────
# CONFIG PAGE
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Lab Biométrie",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CSS PERSONNALISÉ
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0d1117; color: #e6edf3; }
    .main-title { font-size: 2rem; font-weight: 700; color: #58a6ff; margin-bottom: 0.2rem; }
    .sub-title  { font-size: 0.95rem; color: #8b949e; margin-bottom: 1.5rem; }
    .metric-card {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 1rem; text-align: center; margin: 0.3rem 0;
    }
    .status-ok   { color: #3fb950; font-weight: 700; font-size: 1.3rem; }
    .status-fail { color: #f85149; font-weight: 700; font-size: 1.3rem; }
    .info-box {
        background: #161b22; border-left: 4px solid #58a6ff;
        border-radius: 6px; padding: 0.8rem 1rem; margin: 0.5rem 0;
        font-size: 0.9rem; color: #8b949e;
    }
    .hash-box {
        background: #0d1117; border: 1px solid #30363d;
        border-radius: 6px; padding: 0.6rem 1rem;
        font-family: monospace; font-size: 0.8rem; color: #79c0ff;
        word-break: break-all;
    }
    div[data-testid="stButton"] button {
        background: #238636; color: white; border: none;
        border-radius: 6px; font-weight: 600;
        transition: background 0.2s;
    }
    div[data-testid="stButton"] button:hover { background: #2ea043; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "database" not in st.session_state:
    st.session_state.database = {}

database = st.session_state.database

# ─────────────────────────────────────────────
# MEDIAPIPE INIT
# ─────────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection
mp_drawing     = mp.solutions.drawing_utils

# ─────────────────────────────────────────────
# FONCTIONS CŒUR
# ─────────────────────────────────────────────

def pil_to_bgr(pil_img):
    """Convertit une image PIL en tableau BGR pour OpenCV."""
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def get_face_embedding(image_bgr):
    """
    Extrait un embedding facial via MediaPipe FaceMesh.
    Retourne un vecteur normalisé 1404 dimensions (468 landmarks × 3),
    ou None si aucun visage n'est détecté.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.4
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return None

    lm = results.multi_face_landmarks[0].landmark
    vec = np.array([[l.x, l.y, l.z] for l in lm]).flatten()
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8)


def face_detected(image_bgr):
    """Vérifie rapidement si un visage est présent."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    with mp_face_detect.FaceDetection(min_detection_confidence=0.4) as det:
        results = det.process(image_rgb)
    return bool(results.detections)


def enroll_user(name, image_bgr):
    """Enregistre un utilisateur avec son embedding MediaPipe."""
    embedding = get_face_embedding(image_bgr)
    if embedding is not None:
        database[name] = {"template": embedding.tolist()}
        return True, embedding
    return False, None


def recognize_user(image_bgr):
    """Compare l'embedding courant à la base de données (similarité cosinus)."""
    embedding = get_face_embedding(image_bgr)
    if embedding is None:
        return None, 0.0, {}

    scores = {}
    for name, data in database.items():
        stored = np.array(data["template"])
        similarity = float(np.dot(embedding, stored))  # vecteurs déjà normalisés
        scores[name] = round(max(0.0, similarity), 4)

    if not scores:
        return None, 0.0, {}

    best_match = max(scores, key=scores.get)
    return best_match, scores[best_match], scores


def anti_spoofing(image_bgr):
    """
    3 métriques de vivacité :
    - Netteté (Laplacian variance) : photo floue/écran = faible score
    - Texture locale : impressions papier = texture plate
    - Saturation : LCD/impression altèrent les couleurs
    Retourne dict {label: (valeur, seuil, ok)}
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    texture    = float(np.std(gray.astype(np.float32)))
    hsv        = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = float(hsv[:, :, 1].mean())

    return {
        "Netteté":    (round(blur_score, 1), 50,  blur_score > 50),
        "Texture":    (round(texture, 1),    20,  texture > 20),
        "Saturation": (round(saturation, 1), 30,  saturation > 30),
    }


# ─── MÉTHODES DE PROTECTION ───────────────────

def bio_hash(encoding, salt=None):
    """Bio-Hashing : template → vecteur binaire → HMAC-SHA256."""
    if salt is None:
        salt = os.urandom(32)
    rng = np.random.default_rng(int.from_bytes(salt[:8], "big"))
    mat = rng.standard_normal((len(encoding), len(encoding) // 2))
    proj = encoding @ mat
    vec  = (proj > 0).astype(int)
    h    = hmac.new(salt, vec.tobytes(), hashlib.sha256).hexdigest()
    return h, salt, vec


def cancelable_transform(encoding, key):
    """Transformation non-inversible : permutation + seuillage aléatoire."""
    rng  = np.random.default_rng(int.from_bytes(key[:8], "big"))
    perm = rng.permutation(len(encoding))
    threshold = rng.uniform(-0.05, 0.05, len(encoding))
    return np.sign(encoding[perm] - threshold)


def anonymize(encoding):
    """Biométrie anonyme : SHA-256 sans nom stocké."""
    salt = os.urandom(16)
    anon_id = hashlib.sha256(encoding.tobytes() + salt).hexdigest()
    return anon_id, salt


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🔐 Lab Biométrie")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "📸 Enrollment",
    "🔓 Login",
    "🛡️ Protection avancée",
    "📊 Comparatif"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"**👥 Utilisateurs :** {len(database)}")
if database:
    for n in database:
        st.sidebar.markdown(f"- {n}")
if st.sidebar.button("🗑️ Réinitialiser la base"):
    st.session_state.database = {}
    st.rerun()

# ─────────────────────────────────────────────
# PAGE 1 — ENROLLMENT
# ─────────────────────────────────────────────
if page == "📸 Enrollment":
    st.markdown('<div class="main-title">📸 Enregistrement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Capturez votre visage pour l\'enregistrer dans la base biométrique.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        name = st.text_input("Nom de l'utilisateur", placeholder="ex: Yasmine")
        img_file = st.camera_input("📷 Prendre une photo")

        if img_file:
            pil_img = Image.open(img_file)
            image_bgr = pil_to_bgr(pil_img)

            if st.button("✅ Enregistrer", use_container_width=True):
                if not name.strip():
                    st.error("❌ Entrez un nom d'abord")
                else:
                    with st.spinner("Extraction de l'embedding..."):
                        ok, emb = enroll_user(name.strip(), image_bgr)
                    if ok:
                        st.success(f"✅ **{name}** enregistré avec succès !")
                        st.markdown(f'<div class="info-box">Embedding : vecteur 1404 dimensions (468 landmarks × 3 coords) — normalisé L2</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ Aucun visage détecté — réessayez avec un meilleur éclairage")

    with col2:
        st.markdown("### ℹ️ Comment ça fonctionne")
        st.markdown("""
<div class="info-box">
<b>1. Détection</b> — MediaPipe FaceMesh localise 468 points de repère (landmarks) sur votre visage.<br><br>
<b>2. Embedding</b> — Les coordonnées 3D de chaque landmark forment un vecteur de 1404 dimensions.<br><br>
<b>3. Normalisation</b> — Le vecteur est normalisé (norme L2 = 1) pour la comparaison par cosinus.<br><br>
<b>4. Stockage</b> — Seul le vecteur normalisé est conservé en mémoire de session.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — LOGIN
# ─────────────────────────────────────────────
elif page == "🔓 Login":
    st.markdown('<div class="main-title">🔓 Connexion biométrique</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Authentification par reconnaissance faciale avec vérification de vivacité.</div>', unsafe_allow_html=True)

    if not database:
        st.warning("⚠️ Aucun utilisateur enregistré. Allez d'abord sur **Enrollment**.")
    else:
        img_file = st.camera_input("📷 Capturez votre visage")

        if img_file:
            pil_img   = Image.open(img_file)
            image_bgr = pil_to_bgr(pil_img)

            # ── Anti-Spoofing ──
            st.markdown("### 🔍 Vérification de vivacité")
            spoof_scores = anti_spoofing(image_bgr)
            cols = st.columns(3)
            passed = 0
            for i, (label, (val, threshold, ok)) in enumerate(spoof_scores.items()):
                with cols[i]:
                    icon   = "✅" if ok else "❌"
                    status = "status-ok" if ok else "status-fail"
                    st.markdown(f"""
<div class="metric-card">
  <div class="{status}">{icon} {label}</div>
  <div style="font-size:1.4rem;font-weight:700;margin:0.3rem 0">{val}</div>
  <div style="color:#8b949e;font-size:0.8rem">seuil : {threshold}</div>
</div>""", unsafe_allow_html=True)
                if ok:
                    passed += 1

            st.markdown(f"**Score de vivacité : {passed}/3**")

            if passed < 2:
                st.error("🚨 **Attaque possible détectée !** Image suspecte (photo, écran ou masque).")
                st.stop()

            st.success("✅ Vivacité confirmée")
            st.markdown("---")

            # ── Reconnaissance ──
            st.markdown("### 🧠 Reconnaissance faciale")
            with st.spinner("Analyse en cours..."):
                best, score, all_scores = recognize_user(image_bgr)

            if all_scores:
                for name_db, s in sorted(all_scores.items(), key=lambda x: -x[1]):
                    bar_color = "#3fb950" if s > 0.5 else "#8b949e"
                    st.markdown(f"**{name_db}** — {s:.2%}")
                    st.progress(min(s, 1.0))

            st.markdown("---")
            THRESHOLD = 0.50
            if best and score >= THRESHOLD:
                st.markdown(f"""
<div style="background:#0d4429;border:1px solid #3fb950;border-radius:10px;padding:1.2rem;text-align:center">
  <div style="font-size:2rem">✅</div>
  <div style="font-size:1.3rem;font-weight:700;color:#3fb950">Accès accordé</div>
  <div style="color:#e6edf3;margin-top:0.3rem">Bienvenue <b>{best}</b> — Confiance : {score:.2%}</div>
</div>""", unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f"""
<div style="background:#2d1117;border:1px solid #f85149;border-radius:10px;padding:1.2rem;text-align:center">
  <div style="font-size:2rem">❌</div>
  <div style="font-size:1.3rem;font-weight:700;color:#f85149">Accès refusé</div>
  <div style="color:#8b949e;margin-top:0.3rem">Score maximum : {score:.2%} (seuil : {THRESHOLD:.0%})</div>
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3 — PROTECTION AVANCÉE
# ─────────────────────────────────────────────
elif page == "🛡️ Protection avancée":
    st.markdown('<div class="main-title">🛡️ Méthodes de protection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Démonstration interactive des techniques de protection des templates biométriques.</div>', unsafe_allow_html=True)

    img_file = st.camera_input("📷 Image de démonstration")

    if img_file:
        pil_img   = Image.open(img_file)
        image_bgr = pil_to_bgr(pil_img)

        with st.spinner("Extraction de l'embedding..."):
            embedding = get_face_embedding(image_bgr)

        if embedding is None:
            st.error("❌ Aucun visage détecté — réessayez")
        else:
            st.success(f"✅ Embedding extrait — {len(embedding)} dimensions")
            st.markdown("---")

            tab1, tab2, tab3 = st.tabs([
                "🔑 Bio-Hashing",
                "🔄 Cancelable Biometrics",
                "🎭 Biométrie Anonyme"
            ])

            # ── Tab 1 : Bio-Hashing ──
            with tab1:
                st.markdown("### 🔑 Bio-Hashing")
                st.markdown("""
<div class="info-box">
<b>Principe :</b> Le template est projeté sur une matrice pseudo-aléatoire générée par un sel secret,
binarisé, puis hashé via HMAC-SHA256.<br>
<b>Avantage :</b> Même si le hash est volé, il est impossible de reconstruire le visage original.
Si la base est compromise, on change le sel → nouveau hash différent.
</div>""", unsafe_allow_html=True)

                h, salt, vec = bio_hash(embedding)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Hash stocké (HMAC-SHA256)**")
                    st.markdown(f'<div class="hash-box">{h}</div>', unsafe_allow_html=True)
                    st.markdown("**Sel aléatoire (32 octets)**")
                    st.markdown(f'<div class="hash-box">{salt.hex()}</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown("**Vecteur binaire (64 bits)**")
                    bits = "".join(str(b) for b in vec[:64])
                    st.markdown(f'<div class="hash-box">{bits}</div>', unsafe_allow_html=True)
                    st.markdown("""
<div class="info-box" style="margin-top:0.5rem">
✅ Template brut : <b>NON STOCKÉ</b><br>
✅ Révocable : changer le sel<br>
✅ Non-inversible : HMAC one-way
</div>""", unsafe_allow_html=True)

                # Vérification
                st.markdown("**Test de vérification :**")
                h2, _, _ = bio_hash(embedding, salt=salt)
                if h == h2:
                    st.success("✅ Même image + même sel → Hash identique (authentification réussie)")
                else:
                    st.error("❌ Hash différent")

                h3, _, _ = bio_hash(embedding, salt=os.urandom(32))
                st.info(f"🔄 Avec un sel différent → hash totalement différent : `{h3[:32]}...`")

            # ── Tab 2 : Cancelable ──
            with tab2:
                st.markdown("### 🔄 Cancelable Biometrics")
                st.markdown("""
<div class="info-box">
<b>Principe :</b> On applique une transformation F(template, clé) non-inversible et non-réversible.
Si le template est compromis, on change la clé pour générer un nouveau template différent.
</div>""", unsafe_allow_html=True)

                key1 = os.urandom(16)
                key2 = os.urandom(16)
                t1 = cancelable_transform(embedding, key1)
                t2 = cancelable_transform(embedding, key2)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Template avec Clé 1**")
                    st.markdown(f'<div class="hash-box">Clé : {key1.hex()[:16]}...<br>Template : [{", ".join(str(int(x)) for x in t1[:12])}...]</div>', unsafe_allow_html=True)
                with c2:
                    st.markdown("**Template avec Clé 2 (même visage !)**")
                    st.markdown(f'<div class="hash-box">Clé : {key2.hex()[:16]}...<br>Template : [{", ".join(str(int(x)) for x in t2[:12])}...]</div>', unsafe_allow_html=True)

                hamming = float(np.mean(t1 != t2))
                st.markdown(f"""
<div class="info-box">
Distance de Hamming entre les deux templates (même visage, clés différentes) : <b>{hamming:.2%}</b><br>
→ Les templates sont <b>décorrélés</b> : impossible de lier les deux bases de données entre elles.
</div>""", unsafe_allow_html=True)

            # ── Tab 3 : Anonyme ──
            with tab3:
                st.markdown("### 🎭 Biométrie Anonyme")
                st.markdown("""
<div class="info-box">
<b>Principe :</b> Le template est transformé en identifiant anonyme via SHA-256 + sel.
Aucun nom n'est associé dans la base. Un tiers de confiance peut vérifier sans connaître l'identité.
</div>""", unsafe_allow_html=True)

                anon1, salt1 = anonymize(embedding)
                anon2, salt2 = anonymize(embedding)

                st.markdown("**ID anonyme généré**")
                st.markdown(f'<div class="hash-box">{anon1}</div>', unsafe_allow_html=True)
                st.markdown("**Même visage, sel différent → ID totalement différent**")
                st.markdown(f'<div class="hash-box">{anon2}</div>', unsafe_allow_html=True)

                st.markdown("""
<div class="info-box">
✅ Aucun nom stocké dans la base<br>
✅ Lien identité → biométrie impossible sans le sel<br>
✅ Conforme RGPD pour les systèmes à tiers de confiance
</div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — COMPARATIF
# ─────────────────────────────────────────────
elif page == "📊 Comparatif":
    st.markdown('<div class="main-title">📊 Comparatif des méthodes</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Tableau de synthèse des méthodes de protection biométrique.</div>', unsafe_allow_html=True)

    import pandas as pd

    df = pd.DataFrame([
        {
            "Méthode":         "Template brut",
            "Révocable":       "❌",
            "Non-inversible":  "❌",
            "Anonyme":         "❌",
            "Complexité":      "⭐",
            "Usage":           "Démo / prototypage uniquement"
        },
        {
            "Méthode":         "Bio-Hashing",
            "Révocable":       "✅",
            "Non-inversible":  "✅",
            "Anonyme":         "⚠️ Partielle",
            "Complexité":      "⭐⭐",
            "Usage":           "Authentification serveur"
        },
        {
            "Méthode":         "Cancelable Biometrics",
            "Révocable":       "✅",
            "Non-inversible":  "✅",
            "Anonyme":         "⚠️ Partielle",
            "Complexité":      "⭐⭐",
            "Usage":           "Multi-systèmes (clé différente par DB)"
        },
        {
            "Méthode":         "Biométrie Anonyme",
            "Révocable":       "⚠️ Partielle",
            "Non-inversible":  "✅",
            "Anonyme":         "✅",
            "Complexité":      "⭐⭐⭐",
            "Usage":           "RGPD, tiers de confiance"
        },
        {
            "Méthode":         "Fuzzy Vault (crypto)",
            "Révocable":       "⚠️ Partielle",
            "Non-inversible":  "✅",
            "Anonyme":         "✅",
            "Complexité":      "⭐⭐⭐⭐",
            "Usage":           "Haute sécurité + chiffrement"
        },
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 🔐 Recommandations production")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("""
<div class="info-box">
✅ Ne jamais stocker le template brut en base de données<br>
✅ Toujours appliquer une transformation non-inversible<br>
✅ Combiner avec liveness detection (anti-spoofing)<br>
✅ Chiffrer la base avec AES-256 (clé dans un HSM)<br>
✅ Utiliser l'authentification multi-facteur (biométrie + PIN)
</div>""", unsafe_allow_html=True)

    with cols[1]:
        st.markdown("""
<div class="info-box">
📌 <b>Cette app utilise :</b><br>
• MediaPipe FaceMesh — 468 landmarks 3D<br>
• Embedding 1404D normalisé (L2)<br>
• Similarité cosinus pour la comparaison<br>
• Anti-spoofing 3 métriques (blur/texture/sat)<br>
• Bio-Hash + Cancelable + Anonyme en démo
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Session actuelle :** {len(database)} utilisateur(s) enregistré(s)")
    if database:
        st.markdown("**Utilisateurs :** " + ", ".join(database.keys()))
