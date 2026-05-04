import streamlit as st
import numpy as np
import hashlib
import hmac
import os
from PIL import Image
import pandas as pd
from skimage import color, feature, transform, filters, exposure
from scipy.spatial.distance import cosine

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="Lab Biométrie", page_icon="🔐", layout="wide")

st.markdown("""
<style>
    .main-title { font-size:2rem; font-weight:700; color:#58a6ff; margin-bottom:0.2rem; }
    .sub-title  { font-size:0.95rem; color:#8b949e; margin-bottom:1.5rem; }
    .info-box   { background:#161b22; border-left:4px solid #58a6ff; border-radius:6px;
                  padding:0.8rem 1rem; margin:0.5rem 0; font-size:0.9rem; color:#c9d1d9; }
    .hash-box   { background:#0d1117; border:1px solid #30363d; border-radius:6px;
                  padding:0.6rem 1rem; font-family:monospace; font-size:0.8rem;
                  color:#79c0ff; word-break:break-all; margin:0.3rem 0; }
    .result-ok  { background:#0d4429; border:1px solid #3fb950; border-radius:10px;
                  padding:1.2rem; text-align:center; margin:1rem 0; }
    .result-fail{ background:#2d1117; border:1px solid #f85149; border-radius:10px;
                  padding:1.2rem; text-align:center; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
if "database" not in st.session_state:
    st.session_state.database = {}
database = st.session_state.database

# ─────────────────────────────────────────────
# FONCTIONS CŒUR  (pur NumPy / scikit-image)
# ─────────────────────────────────────────────

def pil_to_gray(pil_img):
    """PIL → tableau float64 niveaux de gris [0,1]."""
    return color.rgb2gray(np.array(pil_img.convert("RGB")))


def detect_face_simple(gray):
    """
    Détection de visage simplifiée basée sur la variance locale :
    - divise l'image en régions et trouve la zone la plus 'active'
    - retourne (y, x, h, w) de la meilleure région.
    Pour un vrai Haar Cascade sans cv2, on utilise cette heuristique légère.
    """
    h, w = gray.shape
    # Grille 4×4
    rows, cols = 4, 4
    rh, rw = h // rows, w // cols
    best_var, best_coords = 0, (h//4, w//4, h//2, w//2)
    for r in range(rows):
        for c in range(cols):
            y1, y2 = r*rh, (r+1)*rh
            x1, x2 = c*rw, (c+1)*rw
            region = gray[y1:y2, x1:x2]
            var = float(np.var(region))
            if var > best_var:
                best_var = var
                best_coords = (y1, x1, y2-y1, x2-x1)
    return best_coords  # (y, x, h, w)


def get_embedding(pil_img):
    """
    Embedding facial via HOG (scikit-image) sur ROI 64×64.
    Retourne un vecteur normalisé L2.
    """
    gray = pil_to_gray(pil_img)
    y, x, h, w = detect_face_simple(gray)
    roi  = gray[y:y+h, x:x+w]
    roi  = transform.resize(roi, (64, 64), anti_aliasing=True)

    # HOG descriptor
    fd = feature.hog(
        roi,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        feature_vector=True
    )
    norm = np.linalg.norm(fd)
    return fd / (norm + 1e-8)


def draw_face_box(pil_img):
    """Dessine un rectangle sur le visage détecté, retourne PIL Image."""
    gray = pil_to_gray(pil_img)
    y, x, h, w = detect_face_simple(gray)
    arr = np.array(pil_img.convert("RGB")).copy()
    # Dessiner le rectangle (bleu)
    thickness = 3
    arr[y:y+thickness,   x:x+w]   = [88, 166, 255]
    arr[y+h:y+h+thickness, x:x+w] = [88, 166, 255]
    arr[y:y+h, x:x+thickness]     = [88, 166, 255]
    arr[y:y+h, x+w:x+w+thickness] = [88, 166, 255]
    return Image.fromarray(arr)


def enroll_user(name, pil_img):
    emb = get_embedding(pil_img)
    database[name] = {"template": emb.tolist()}
    return emb


def recognize_user(pil_img):
    emb = get_embedding(pil_img)
    scores = {}
    for name, data in database.items():
        stored = np.array(data["template"])
        sim    = float(np.dot(emb, stored))
        scores[name] = round(max(0.0, sim), 4)
    if not scores:
        return None, 0.0, {}
    best = max(scores, key=scores.get)
    return best, scores[best], scores


def anti_spoofing(pil_img):
    gray  = pil_to_gray(pil_img)
    # Netteté : Laplacian variance
    lap   = filters.laplace(gray)
    blur  = round(float(np.var(lap) * 1000), 1)
    # Texture : écart-type local
    tex   = round(float(np.std(gray) * 100), 1)
    # Saturation
    rgb   = np.array(pil_img.convert("RGB")).astype(float) / 255.0
    cmax  = rgb.max(axis=2)
    cmin  = rgb.min(axis=2)
    sat   = round(float(np.mean(np.where(cmax == 0, 0, (cmax - cmin) / (cmax + 1e-8)))) * 100, 1)
    return {
        "Netteté":    (blur, 5,  blur > 5),
        "Texture":    (tex,  5,  tex  > 5),
        "Saturation": (sat,  10, sat  > 10),
    }


# ── PROTECTION ─────────────────────────────────────────────────

def bio_hash(encoding, salt=None):
    if salt is None:
        salt = os.urandom(32)
    rng  = np.random.default_rng(int.from_bytes(salt[:8], "big"))
    mat  = rng.standard_normal((len(encoding), len(encoding) // 2))
    vec  = (encoding @ mat > 0).astype(int)
    h    = hmac.new(salt, vec.tobytes(), hashlib.sha256).hexdigest()
    return h, salt, vec


def cancelable_transform(encoding, key):
    rng  = np.random.default_rng(int.from_bytes(key[:8], "big"))
    perm = rng.permutation(len(encoding))
    return np.sign(encoding[perm] - rng.uniform(-0.05, 0.05, len(encoding)))


def anonymize(encoding):
    salt = os.urandom(16)
    return hashlib.sha256(encoding.tobytes() + salt).hexdigest(), salt


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
st.sidebar.markdown("## 🔐 Lab Biométrie")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    " Enrollment", " Login", " Protection avancée", " Comparatif"
])
st.sidebar.markdown("---")
st.sidebar.markdown(f"** Utilisateurs :** {len(database)}")
for n in database:
    st.sidebar.markdown(f"- {n}")
if st.sidebar.button(" Réinitialiser"):
    st.session_state.database = {}
    st.rerun()

# ─────────────────────────────────────────────
# PAGE 1 — ENROLLMENT
# ─────────────────────────────────────────────
if page == "📸 Enrollment":
    st.markdown('<div class="main-title"> Enregistrement</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Capturez votre visage pour l\'enregistrer dans la base biométrique.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        name     = st.text_input("Nom", placeholder="ex: Yasmine")
        img_file = st.camera_input("📷 Prendre une photo")

        if img_file:
            pil_img = Image.open(img_file)
            st.image(draw_face_box(pil_img), caption="Détection ROI", use_container_width=True)

            if st.button(" Enregistrer", use_container_width=True):
                if not name.strip():
                    st.error("❌ Entrez un nom d'abord")
                else:
                    emb = enroll_user(name.strip(), pil_img)
                    st.success(f" **{name}** enregistré ! Embedding HOG : {len(emb)} dimensions")

    with col2:
        st.markdown("###  Pipeline")
        st.markdown("""
<div class="info-box">
<b>1. ROI</b> — Sélection de la zone la plus active (variance max).<br><br>
<b>2. Resize 64×64</b> — Normalisation de la taille.<br><br>
<b>3. HOG descriptor</b> — Histogram of Oriented Gradients via scikit-image
(cellules 8×8, blocs 2×2, 9 orientations).<br><br>
<b>4. Normalisation L2</b> — Vecteur prêt pour similarité cosinus.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2 — LOGIN
# ─────────────────────────────────────────────
elif page == " Login":
    st.markdown('<div class="main-title"> Connexion biométrique</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Authentification faciale + vérification de vivacité.</div>', unsafe_allow_html=True)

    if not database:
        st.warning(" Aucun utilisateur enregistré. Allez d'abord sur **Enrollment**.")
    else:
        img_file = st.camera_input(" Capturez votre visage")
        if img_file:
            pil_img = Image.open(img_file)
            st.image(draw_face_box(pil_img), caption="Détection", use_container_width=True)

            # Anti-Spoofing
            st.markdown("###  Vivacité")
            spoof  = anti_spoofing(pil_img)
            cols   = st.columns(3)
            passed = 0
            for i, (label, (val, thr, ok)) in enumerate(spoof.items()):
                with cols[i]:
                    st.metric(f"{'✅' if ok else '❌'} {label}", val, f"seuil {thr}")
                if ok:
                    passed += 1

            st.markdown(f"**Score : {passed}/3**")
            if passed < 2:
                st.error(" Image suspecte — possible attaque !")
                st.stop()
            st.success(" Vivacité confirmée")
            st.markdown("---")

            # Reconnaissance
            st.markdown("###  Reconnaissance")
            best, score, all_scores = recognize_user(pil_img)
            for n, s in sorted(all_scores.items(), key=lambda x: -x[1]):
                st.markdown(f"**{n}** — {s:.2%}")
                st.progress(min(float(s), 1.0))

            st.markdown("---")
            THRESHOLD = 0.6
            if best and score >= THRESHOLD:
                st.markdown(f'<div class="result-ok"><div style="font-size:2rem">✅</div><div style="font-size:1.3rem;font-weight:700;color:#3fb950">Accès accordé</div><div style="color:#e6edf3">Bienvenue <b>{best}</b> — {score:.2%}</div></div>', unsafe_allow_html=True)
                st.balloons()
            else:
                st.markdown(f'<div class="result-fail"><div style="font-size:2rem">❌</div><div style="font-size:1.3rem;font-weight:700;color:#f85149">Accès refusé</div><div style="color:#8b949e">Score : {score:.2%} — seuil : {THRESHOLD:.0%}</div></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 3 — PROTECTION
# ─────────────────────────────────────────────
elif page == " Protection avancée":
    st.markdown('<div class="main-title"> Méthodes de protection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Démonstration des techniques de protection des templates biométriques.</div>', unsafe_allow_html=True)

    img_file = st.camera_input("📷 Image de démonstration")
    if img_file:
        pil_img = Image.open(img_file)
        emb = get_embedding(pil_img)
        st.success(f" Embedding HOG extrait — {len(emb)} dimensions")
        st.markdown("---")

        tab1, tab2, tab3 = st.tabs(["🔑 Bio-Hashing", "🔄 Cancelable", "🎭 Anonyme"])

        with tab1:
            st.markdown("###  Bio-Hashing")
            st.markdown('<div class="info-box">Projection sur matrice pseudo-aléatoire (sel secret) → vecteur binaire → HMAC-SHA256. Le template brut n\'est jamais stocké. Si compromis : changer le sel → nouveau hash.</div>', unsafe_allow_html=True)
            h, salt, vec = bio_hash(emb)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Hash HMAC-SHA256**")
                st.markdown(f'<div class="hash-box">{h}</div>', unsafe_allow_html=True)
                st.markdown("**Sel (32 octets)**")
                st.markdown(f'<div class="hash-box">{salt.hex()}</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**Vecteur binaire (64 bits)**")
                st.markdown(f'<div class="hash-box">{"".join(str(b) for b in vec[:64])}</div>', unsafe_allow_html=True)
                st.markdown('<div class="info-box"> Template brut : NON STOCKÉ<br> Révocable<br> Non-inversible</div>', unsafe_allow_html=True)
            h2, _, _ = bio_hash(emb, salt=salt)
            st.success(" Même sel → hash identique (auth OK)") if h == h2 else st.error("❌")
            h3, _, _ = bio_hash(emb, salt=os.urandom(32))
            st.info(f" Sel différent → `{h3[:32]}...`")

        with tab2:
            st.markdown("###  Cancelable Biometrics")
            st.markdown('<div class="info-box">Permutation + seuillage par clé secrète. Même visage + clé différente = template totalement différent. Révocable en changeant la clé.</div>', unsafe_allow_html=True)
            k1, k2 = os.urandom(16), os.urandom(16)
            t1, t2 = cancelable_transform(emb, k1), cancelable_transform(emb, k2)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Clé 1**")
                st.markdown(f'<div class="hash-box">{k1.hex()[:16]}...<br>[{", ".join(str(int(x)) for x in t1[:12])}...]</div>', unsafe_allow_html=True)
            with c2:
                st.markdown("**Clé 2 (même visage)**")
                st.markdown(f'<div class="hash-box">{k2.hex()[:16]}...<br>[{", ".join(str(int(x)) for x in t2[:12])}...]</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="info-box">Distance de Hamming : <b>{float(np.mean(t1!=t2)):.2%}</b> — templates décorrélés </div>', unsafe_allow_html=True)

        with tab3:
            st.markdown("###  Biométrie Anonyme")
            st.markdown('<div class="info-box">SHA-256(template + sel) → identifiant anonyme. Aucun nom stocké. Conforme RGPD.</div>', unsafe_allow_html=True)
            a1, _ = anonymize(emb)
            a2, _ = anonymize(emb)
            st.markdown("**ID — Sel 1**")
            st.markdown(f'<div class="hash-box">{a1}</div>', unsafe_allow_html=True)
            st.markdown("**ID — Sel 2 (même visage)**")
            st.markdown(f'<div class="hash-box">{a2}</div>', unsafe_allow_html=True)
            st.markdown('<div class="info-box"> Aucun nom stocké<br> Lien identité → biométrie impossible sans le sel</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 4 — COMPARATIF
# ─────────────────────────────────────────────
elif page == " Comparatif":
    st.markdown('<div class="main-title">📊 Comparatif</div>', unsafe_allow_html=True)
    df = pd.DataFrame([
        {"Méthode":"Template brut",        "Révocable":"❌","Non-inversible":"❌","Anonyme":"❌", "Complexité":"⭐",     "Usage":"Démo uniquement"},
        {"Méthode":"Bio-Hashing",          "Révocable":"✅","Non-inversible":"✅","Anonyme":"⚠️","Complexité":"⭐⭐",    "Usage":"Auth serveur"},
        {"Méthode":"Cancelable",           "Révocable":"✅","Non-inversible":"✅","Anonyme":"⚠️","Complexité":"⭐⭐",    "Usage":"Multi-systèmes"},
        {"Méthode":"Biométrie Anonyme",    "Révocable":"⚠️","Non-inversible":"✅","Anonyme":"✅", "Complexité":"⭐⭐⭐",   "Usage":"RGPD"},
        {"Méthode":"Fuzzy Vault (crypto)", "Révocable":"⚠️","Non-inversible":"✅","Anonyme":"✅", "Complexité":"⭐⭐⭐⭐",  "Usage":"Haute sécurité"},
    ])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("###  Recommandations")
        st.markdown('<div class="info-box"> Ne jamais stocker le template brut<br> Transformation non-inversible obligatoire<br> Liveness detection combinée<br> Chiffrement AES-256 + HSM<br> Multi-facteur : biométrie + PIN</div>', unsafe_allow_html=True)
    with c2:
        st.markdown("###  Stack de cette app")
        st.markdown('<div class="info-box">• Détection : variance locale (ROI)<br>• Embedding : HOG (scikit-image)<br>• Comparaison : similarité cosinus<br>• Anti-spoofing : Laplacian + texture + sat.<br>• Protection : Bio-Hash + Cancelable + Anonyme</div>', unsafe_allow_html=True)
    st.info(f"Session : {len(database)} utilisateur(s) — {', '.join(database.keys()) if database else 'aucun'}")
