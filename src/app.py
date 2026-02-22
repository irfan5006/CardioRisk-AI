import os
import streamlit as st
import pandas as pd
import joblib

# ──────────────────────────────────────────────────────────────────
# Base directory (project root = one level up from src/)
# ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ──────────────────────────────────────────────────────────────────
# Page Config — must be the very first Streamlit command
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioVision AI — Heart Disease Risk Prediction",
    page_icon="🫀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────
# Custom CSS — Clean, minimal, medical‑themed on WHITE background
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@600;700;800&display=swap');

/* ── Root Design Tokens ── */
:root {
    --white:         #FFFFFF;
    --bg:            #F7F9FC;
    --primary:       #1B6B93;
    --primary-dark:  #155A7A;
    --primary-light: #D6EAF8;
    --accent:        #2E86AB;
    --text-dark:     #1A1A2E;
    --text-body:     #2D3436;
    --text-muted:    #636E72;
    --border:        #E0E7EF;
    --danger:        #D63031;
    --danger-bg:     #FDECEA;
    --danger-border: #F5C6CB;
    --success:       #00B894;
    --success-bg:    #E8F8F5;
    --success-border:#B2DFDB;
    --shadow-sm:     0 1px 4px rgba(0,0,0,.06);
    --shadow-md:     0 4px 16px rgba(27,107,147,.10);
    --shadow-lg:     0 8px 32px rgba(27,107,147,.12);
    --radius:        12px;
    --radius-sm:     8px;
    --transition:    all .25s cubic-bezier(.4,0,.2,1);
}

/* ── Global Background ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-body);
}

/* ── Hide Streamlit Defaults ── */
header[data-testid="stHeader"]  { background: transparent !important; }
footer                          { display: none !important; }
#MainMenu                       { visibility: hidden !important; }
.stDeployButton                 { display: none !important; }

/* ── Centered Container Max Width ── */
.block-container {
    max-width: 820px !important;
    padding: 2rem 1.5rem 3rem !important;
}

/* ══════════════════════════════════════════
   HEADER BANNER
   ══════════════════════════════════════════ */
.hero-header {
    background: linear-gradient(135deg, #1B6B93 0%, #2E86AB 50%, #3AAFA9 100%);
    border-radius: var(--radius);
    padding: 2.8rem 2rem;
    text-align: center;
    color: #fff;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
    margin-bottom: 1.8rem;
}
.hero-header::before {
    content: '';
    position: absolute;
    width: 300px; height: 300px;
    border-radius: 50%;
    background: rgba(255,255,255,.06);
    top: -100px; right: -60px;
}
.hero-header::after {
    content: '';
    position: absolute;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,.04);
    bottom: -80px; left: -40px;
}
.hero-header .hero-icon {
    font-size: 3rem;
    margin-bottom: .4rem;
    display: block;
    filter: drop-shadow(0 2px 8px rgba(0,0,0,.15));
}
.hero-header h1 {
    font-family: 'Poppins', sans-serif;
    font-weight: 800;
    font-size: 2rem;
    margin: 0;
    letter-spacing: .3px;
    line-height: 1.25;
}
.hero-header .subtitle {
    font-weight: 300;
    font-size: 1.05rem;
    opacity: .92;
    margin: .4rem 0 0;
}
.hero-header .tech-badge {
    display: inline-block;
    margin-top: .85rem;
    background: rgba(255,255,255,.15);
    backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,.2);
    padding: 5px 16px;
    border-radius: 20px;
    font-size: .8rem;
    letter-spacing: .4px;
    font-weight: 500;
}

/* ══════════════════════════════════════════
   CARDS
   ══════════════════════════════════════════ */
.card {
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    margin-bottom: 1.4rem;
    box-shadow: var(--shadow-sm);
    transition: var(--transition);
}
.card:hover {
    box-shadow: var(--shadow-md);
    transform: translateY(-1px);
}
.card-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
    padding-bottom: .8rem;
    border-bottom: 2px solid var(--primary-light);
}
.card-title .card-icon {
    width: 40px; height: 40px;
    border-radius: var(--radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    flex-shrink: 0;
    background: var(--primary-light);
    color: var(--primary);
}
.card-title h3 {
    font-family: 'Poppins', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0;
    color: var(--text-dark);
}

/* ── Info Alert ── */
.info-alert {
    background: var(--primary-light);
    border-left: 4px solid var(--primary);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: .75rem 1.2rem;
    color: var(--primary-dark);
    font-size: .9rem;
    font-weight: 500;
    margin-bottom: 1.4rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Section Sub-headers ── */
.section-label {
    font-family: 'Poppins', sans-serif;
    font-size: .92rem;
    font-weight: 600;
    color: var(--text-dark);
    margin-bottom: .6rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

/* ══════════════════════════════════════════
   STREAMLIT WIDGET OVERRIDES
   ══════════════════════════════════════════ */

/* Labels — dark, readable */
[data-testid="stSlider"]      label,
[data-testid="stSelectbox"]   label,
[data-testid="stNumberInput"] label {
    font-weight: 500 !important;
    color: var(--text-dark) !important;
    font-size: .9rem !important;
}

/* Select dropdowns */
div[data-baseweb="select"] > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    background: var(--white) !important;
    color: var(--text-body) !important;
}

/* Number inputs */
input[type="number"] {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
    color: var(--text-body) !important;
    background: var(--white) !important;
}

/* Slider thumb */
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--primary) !important;
}

/* ══════════════════════════════════════════
   PREDICT BUTTON
   ══════════════════════════════════════════ */
.stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: .8rem 2.6rem !important;
    font-family: 'Poppins', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    letter-spacing: .4px;
    box-shadow: 0 4px 14px rgba(27,107,147,.28) !important;
    transition: var(--transition) !important;
    width: 100%;
    cursor: pointer;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 24px rgba(27,107,147,.35) !important;
    filter: brightness(1.05);
}
.stButton > button:active {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(27,107,147,.22) !important;
}

/* ══════════════════════════════════════════
   RESULT ALERT BOXES
   ══════════════════════════════════════════ */
.alert-box {
    border-radius: var(--radius);
    padding: 1.8rem 2rem;
    text-align: center;
    margin-top: .8rem;
    box-shadow: var(--shadow-md);
    animation: fadeSlideIn .5s ease;
}
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.alert-box.danger {
    background: var(--danger-bg);
    border: 1.5px solid var(--danger-border);
}
.alert-box.success {
    background: var(--success-bg);
    border: 1.5px solid var(--success-border);
}
.alert-box .alert-icon {
    font-size: 2.8rem;
    margin-bottom: .3rem;
}
.alert-box h2 {
    font-family: 'Poppins', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    margin: .3rem 0 .2rem;
}
.alert-box.danger  h2 { color: var(--danger); }
.alert-box.success h2 { color: var(--success); }
.alert-box p {
    color: var(--text-body);
    font-size: .92rem;
    margin: 0;
}

/* ── Probability Gauges ── */
.prob-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    margin-bottom: .8rem;
}
.prob-card {
    flex: 1;
    background: var(--white);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: var(--shadow-sm);
}
.prob-card .prob-label {
    font-size: .78rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .7px;
    color: var(--text-muted);
    margin-bottom: .35rem;
}
.prob-card .prob-value {
    font-family: 'Poppins', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
}
.prob-card .prob-value.green { color: var(--success); }
.prob-card .prob-value.red   { color: var(--danger); }
.prob-bar-track {
    height: 6px;
    border-radius: 3px;
    background: #EDF2F7;
    margin-top: .6rem;
    overflow: hidden;
}
.prob-bar-track .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width .7s ease;
}
.prob-bar-track .bar-fill.green { background: var(--success); }
.prob-bar-track .bar-fill.red   { background: var(--danger); }

/* ── Disclaimer ── */
.disclaimer {
    background: #FFF9E6;
    border-left: 4px solid #F0C929;
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    padding: .7rem 1.1rem;
    font-size: .84rem;
    color: #6B5900;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ── Footer ── */
.footer {
    text-align: center;
    margin-top: 2.5rem;
    padding: 1.2rem 0;
    font-size: .82rem;
    color: var(--text-muted);
    border-top: 1px solid var(--border);
}
.footer a {
    color: var(--primary);
    text-decoration: none;
    font-weight: 600;
}
.footer a:hover { text-decoration: underline; }

/* ── Responsive ── */
@media (max-width: 640px) {
    .hero-header { padding: 1.8rem 1.2rem; }
    .hero-header h1 { font-size: 1.5rem; }
    .card { padding: 1.2rem 1rem; }
    .prob-row { flex-direction: column; }
    .block-container { padding: 1rem 1rem 2rem !important; }
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────
# Load Model Artifacts
# ──────────────────────────────────────────────────────────────────
model            = joblib.load(os.path.join(BASE_DIR, "models", "KNN_heart.pkl"))
scaler           = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))
expected_columns = joblib.load(os.path.join(BASE_DIR, "models", "columns.pkl"))

# ══════════════════════════════════════════════════════════════════
# HEADER BANNER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
    <span class="hero-icon">🫀</span>
    <h1>CardioVision</h1>
    <p class="subtitle">Heart Disease Prediction</p>
    <span class="tech-badge">🔬 KNN Machine Learning Model</span>
</div>
""", unsafe_allow_html=True)

# Info alert
st.markdown("""
<div class="info-alert">
    ℹ️&nbsp; Enter the patient's clinical parameters below to generate a risk assessment.
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PREDICTION FORM — Centered Card with Two‑Column Layout
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="card">
    <div class="card-title">
        <div class="card-icon">🩺</div>
        <h3>Patient Clinical Details</h3>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Two‑Column Inputs ──
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="section-label">📋 Demographics & Vitals</div>',
                unsafe_allow_html=True)
    age         = st.slider("Age", 18, 100, 40)
    sex         = st.selectbox("Gender", ["M — Male", "F — Female"])
    chest_pain  = st.selectbox("Chest Pain Type",
                               ["ATA — Atypical Angina",
                                "NAP — Non-Anginal Pain",
                                "TA — Typical Angina",
                                "ASY — Asymptomatic"])
    resting_bp  = st.number_input("Resting Blood Pressure (mm Hg)",
                                  min_value=80, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol (mg/dL)",
                                  min_value=100, max_value=600, value=200)
    fasting_bs  = st.selectbox("Fasting Blood Sugar > 120 mg/dL",
                               ["0 — No", "1 — Yes"])

with col2:
    st.markdown('<div class="section-label">📈 Cardiac Indicators</div>',
                unsafe_allow_html=True)
    resting_ecg     = st.selectbox("Resting ECG",
                                   ["Normal", "ST — ST‑T Wave Abnormality",
                                    "LVH — Left Ventricular Hypertrophy"])
    max_hr          = st.slider("Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise‑Induced Angina",
                                   ["N — No", "Y — Yes"])
    oldpeak         = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0, 0.1)
    st_slope        = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ── Extract short codes from the descriptive labels ──
sex_code         = sex.split(" — ")[0]
chest_code       = chest_pain.split(" — ")[0]
fasting_code     = int(fasting_bs.split(" — ")[0])
ecg_code         = resting_ecg.split(" — ")[0] if " — " in resting_ecg else resting_ecg
angina_code      = exercise_angina.split(" — ")[0]

# ══════════════════════════════════════════════════════════════════
# PREDICT BUTTON
# ══════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍  Predict Heart Disease Risk"):

    # ── Build one‑hot feature dictionary ──
    raw_input = {
        'Age':           age,
        'RestingBP':     resting_bp,
        'Cholesterol':   cholesterol,
        'FastingBS':     fasting_code,
        'MaxHR':         max_hr,
        'Oldpeak':       oldpeak,
        'Sex_' + sex_code:                        1,
        'ChestPainType_' + chest_code:            1,
        'RestingECG_' + ecg_code:                 1,
        'ExerciseAngina_' + angina_code:          1,
        'ST_Slope_' + st_slope:                   1,
    }

    input_df = pd.DataFrame([raw_input])

    # Fill missing one-hot columns with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[expected_columns]

    # Scale & predict
    scaled_input = scaler.transform(input_df)
    prediction   = model.predict(scaled_input)[0]
    probability  = model.predict_proba(scaled_input)[0]
    low_pct      = round(probability[0] * 100, 2)
    high_pct     = round(probability[1] * 100, 2)

    # ── Results Card ──
    st.markdown("""
    <div class="card">
        <div class="card-title">
            <div class="card-icon">📊</div>
            <h3>Risk Analysis Results</h3>
        </div>
    """, unsafe_allow_html=True)

    # Probability gauges
    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-card">
            <div class="prob-label">Low Risk</div>
            <div class="prob-value green">{low_pct}%</div>
            <div class="prob-bar-track">
                <div class="bar-fill green" style="width:{low_pct}%"></div>
            </div>
        </div>
        <div class="prob-card">
            <div class="prob-label">High Risk</div>
            <div class="prob-value red">{high_pct}%</div>
            <div class="prob-bar-track">
                <div class="bar-fill red" style="width:{high_pct}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Verdict alert box
    if prediction == 1:
        st.markdown("""
        <div class="alert-box danger">
            <div class="alert-icon">⚠️</div>
            <h2>High Risk Detected</h2>
            <p>The model indicates an elevated risk of heart disease. Please consult a physician immediately.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="alert-box success">
            <div class="alert-icon">✅</div>
            <h2>Low Risk — No Significant Indicators</h2>
            <p>The model does not indicate a significant risk of heart disease at this time.</p>
        </div>
        """, unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        ⚕️&nbsp; <strong>Disclaimer:</strong>&nbsp; This tool is for educational purposes only
        and is not a substitute for professional medical advice, diagnosis, or treatment.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # close .card

# ══════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    Developed with ❤️ by <strong>Muhammad Irfan</strong>
    &nbsp;|&nbsp; Machine Learning Project
    &nbsp;|&nbsp; Built with <a href="https://streamlit.io" target="_blank">Streamlit</a>
</div>
""", unsafe_allow_html=True)