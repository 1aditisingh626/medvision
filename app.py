import os
import json
import tempfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"

from medical_report_analyzer import (
    parse_and_analyze_report,
    detect_findings,
    extract_values,
    extract_text_from_file,
)

# ============================================================
# MediVision — Multimodal Health Suite
# Polished UI + exact checkpoint handling
# ============================================================

st.set_page_config(
    page_title="MediVision | Multimodal Health Suite",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# PREMIUM UI
# ============================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

:root {
    --navy:#07111f;
    --navy2:#0d1b2e;
    --blue:#2563eb;
    --cyan:#06b6d4;
    --violet:#7c3aed;
    --green:#10b981;
    --red:#ef4444;
    --amber:#f59e0b;
    --text:#102033;
    --muted:#64748b;
    --line:#dbe5f0;
    --surface:#ffffff;
    --soft:#f4f8fc;
}

html, body, [class*="css"] {
    font-family:"DM Sans",sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 92% 0%, rgba(37,99,235,.12), transparent 25%),
        radial-gradient(circle at 5% 20%, rgba(6,182,212,.09), transparent 24%),
        linear-gradient(180deg,#f8fbff 0%,#f3f6fa 100%);
}

.block-container {
    max-width:1480px;
    padding-top:1.2rem;
    padding-bottom:3rem;
}

[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at 80% 5%, rgba(6,182,212,.13), transparent 24%),
        linear-gradient(180deg,#06101d 0%,#0b1728 100%);
    border-right:1px solid rgba(255,255,255,.08);
}

[data-testid="stSidebar"] * {
    color:#e8f1fb !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:rgba(255,255,255,.07) !important;
    border:1px solid rgba(255,255,255,.12) !important;
}

[data-testid="stSidebar"] .stRadio label {
    font-weight:650;
}

.hero {
    position:relative;
    overflow:hidden;
    border-radius:30px;
    padding:38px 42px;
    margin-bottom:26px;
    color:white;
    background:
        radial-gradient(circle at 85% 18%,rgba(6,182,212,.30),transparent 28%),
        radial-gradient(circle at 65% 110%,rgba(124,58,237,.24),transparent 32%),
        linear-gradient(120deg,#06101d 0%,#0c2340 55%,#102d4a 100%);
    box-shadow:0 28px 70px rgba(15,23,42,.20);
}

.hero:after {
    content:"";
    position:absolute;
    width:310px;
    height:310px;
    right:-105px;
    top:-145px;
    border:1px solid rgba(255,255,255,.13);
    border-radius:50%;
    box-shadow:
        0 0 0 38px rgba(255,255,255,.025),
        0 0 0 76px rgba(255,255,255,.018);
}

.hero-kicker {
    display:inline-flex;
    align-items:center;
    gap:9px;
    padding:8px 12px;
    border:1px solid rgba(255,255,255,.14);
    background:rgba(255,255,255,.07);
    border-radius:999px;
    font-size:.76rem;
    font-weight:800;
    letter-spacing:.09em;
    text-transform:uppercase;
}

.dot {
    width:8px;
    height:8px;
    border-radius:50%;
    background:#34d399;
    box-shadow:0 0 0 5px rgba(52,211,153,.13),0 0 18px rgba(52,211,153,.7);
}

.hero h1 {
    font-family:"Manrope",sans-serif;
    font-size:clamp(2.15rem,4vw,3.7rem);
    line-height:1.01;
    margin:19px 0 12px;
    letter-spacing:-.055em;
}

.hero p {
    color:#c9d8e9;
    max-width:850px;
    font-size:1.02rem;
    line-height:1.65;
    margin:0;
}

.section-title {
    font-family:"Manrope",sans-serif;
    color:var(--text);
    font-size:1.65rem;
    font-weight:800;
    letter-spacing:-.035em;
    margin:10px 0 4px;
}

.section-subtitle {
    color:var(--muted);
    margin-bottom:18px;
    line-height:1.55;
}

.panel {
    background:rgba(255,255,255,.90);
    border:1px solid rgba(148,163,184,.23);
    border-radius:22px;
    padding:21px;
    box-shadow:0 12px 35px rgba(15,23,42,.065);
    margin-bottom:18px;
    backdrop-filter:blur(12px);
}

.panel:hover {
    box-shadow:0 17px 42px rgba(15,23,42,.09);
}

.panel-title {
    display:flex;
    align-items:center;
    gap:10px;
    font-family:"Manrope",sans-serif;
    font-size:1.1rem;
    font-weight:800;
    color:var(--text);
    margin-bottom:8px;
}

.status-pill {
    display:inline-flex;
    align-items:center;
    gap:7px;
    border-radius:999px;
    padding:7px 11px;
    font-size:.75rem;
    font-weight:800;
    background:#ecfdf5;
    color:#047857;
    border:1px solid #a7f3d0;
}

.metric {
    background:linear-gradient(145deg,#fff,#f8fbff);
    border:1px solid var(--line);
    border-radius:18px;
    padding:18px;
    min-height:110px;
    box-shadow:0 8px 25px rgba(15,23,42,.045);
}

.metric-label {
    color:var(--muted);
    font-size:.74rem;
    font-weight:800;
    text-transform:uppercase;
    letter-spacing:.07em;
}

.metric-value {
    color:var(--text);
    font-family:"Manrope",sans-serif;
    font-size:1.55rem;
    font-weight:800;
    margin-top:7px;
}

.result-normal,.result-abnormal,.result-info,.result-warning {
    border-radius:18px;
    padding:18px 20px;
    margin:12px 0;
}

.result-normal {
    background:linear-gradient(135deg,#effdf7,#e9fbf3);
    border:1px solid #b7efd6;
    border-left:6px solid #10b981;
}

.result-abnormal {
    background:linear-gradient(135deg,#fff7f7,#fff0f0);
    border:1px solid #fecaca;
    border-left:6px solid #ef4444;
}

.result-info {
    background:linear-gradient(135deg,#f2f8ff,#eef6ff);
    border:1px solid #bfdbfe;
    border-left:6px solid #2563eb;
}

.result-warning {
    background:linear-gradient(135deg,#fffaf0,#fff6e6);
    border:1px solid #fde68a;
    border-left:6px solid #f59e0b;
}

.scan-frame {
    background:#050d18;
    border:1px solid #1d4167;
    border-radius:19px;
    padding:10px;
    box-shadow:inset 0 0 0 1px rgba(255,255,255,.035),0 14px 32px rgba(7,17,31,.14);
}

.confidence-box {
    background:linear-gradient(135deg,#f7f4ff,#fbf7ff);
    border:1px solid #ddd6fe;
    border-left:5px solid #7c3aed;
    border-radius:17px;
    padding:17px 19px;
    margin-top:14px;
}

.footer-note {
    text-align:center;
    color:#64748b;
    font-size:.78rem;
    margin-top:32px;
    padding-top:22px;
    border-top:1px solid #dfe7f0;
}

.stButton > button {
    min-height:48px !important;
    border:0 !important;
    border-radius:13px !important;
    color:#fff !important;
    font-weight:800 !important;
    letter-spacing:.01em;
    background:linear-gradient(135deg,#2563eb 0%,#0891b2 100%) !important;
    box-shadow:0 11px 26px rgba(37,99,235,.22) !important;
    transition:all .18s ease !important;
}

.stButton > button:hover {
    transform:translateY(-2px);
    box-shadow:0 16px 34px rgba(37,99,235,.30) !important;
}

.stDownloadButton > button {
    min-height:45px !important;
    border-radius:12px !important;
    font-weight:800 !important;
}

[data-testid="stFileUploaderDropzone"] {
    border-radius:16px !important;
    border:1.5px dashed #8db9e9 !important;
    background:linear-gradient(145deg,#f8fbff,#f2f8ff) !important;
}

[data-testid="stMetric"] {
    background:#fff;
    border:1px solid var(--line);
    padding:14px;
    border-radius:16px;
    box-shadow:0 7px 22px rgba(15,23,42,.04);
}

[data-testid="stDataFrame"] {
    border-radius:15px;
    overflow:hidden;
}

div[data-baseweb="tab-list"] {
    gap:7px;
}

button[data-baseweb="tab"] {
    border-radius:10px !important;
}

@media(max-width:900px) {
    .hero {padding:26px 23px;border-radius:22px;}
    .panel {padding:16px;}
}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================
def hero():
    st.markdown(
        """
        <section class="hero">
            <div class="hero-kicker"><span class="dot"></span> MediVision • Multimodal Health Suite</div>
            <h1>Intelligent medical insights,<br>designed for clarity.</h1>
            <p>
                Computer-vision screening, renal image classification, and multilingual
                laboratory-report interpretation in one professional workspace.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def panel_title(icon, title, subtitle=None):
    st.markdown(
        f"""
        <div class="panel-title">{icon} {title}</div>
        {f'<div class="section-subtitle">{subtitle}</div>' if subtitle else ''}
        """,
        unsafe_allow_html=True,
    )


def confidence_explanation(confidence_val, pred_class, model_name):
    with st.expander("🧠 Understand the confidence score"):
        st.markdown(
            f"""
**Model:** {model_name}

- **Predicted result:** `{pred_class}`
- **Displayed confidence:** `{confidence_val:.2f}%`
- For classification, the value is derived from softmax probabilities.
- For segmentation, the displayed value is the peak sigmoid activation and is only a visualization signal.
- A high model score does **not** mean a confirmed medical diagnosis.
- Image quality, acquisition protocol, training data and class overlap can affect confidence.
"""
        )


def safe_torch_load(path):
    return torch.load(
        path,
        map_location=torch.device("cpu"),
        weights_only=False,
    )


# ============================================================
# CHEST MODEL
# ============================================================
@st.cache_resource(show_spinner=False)
def load_chest_model(model_path):
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    try:
        checkpoint = safe_torch_load(model_path)

        if not isinstance(checkpoint, dict):
            return None, "Chest checkpoint is not a supported checkpoint dictionary."

        state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict"))
        if state_dict is None:
            return None, "Chest checkpoint does not contain model_state_dict/state_dict."

        class_names = checkpoint.get(
            "class_names",
            ["Infiltration", "Effusion", "Atelectasis"],
        )

        # Remove DataParallel prefix if present.
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(
            model.classifier.in_features,
            len(class_names),
        )

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        return (model, class_names), None

    except Exception as exc:
        return None, f"Unable to load the chest X-ray model: {exc}"


# ============================================================
# KIDNEY FOUR-CLASS KERAS CLASSIFIER
# Class order and preprocessing are verified from kidney_c.ipynb.
# ============================================================
KIDNEY_CLASS_NAMES = ["Cyst", "Normal", "Stone", "Tumor"]


@st.cache_resource(show_spinner=False)
def load_kidney_classifier(model_path, config_path):
    """Load the Keras model only when requested; fail closed without verified config."""
    model_file = Path(model_path)
    config_file = Path(config_path)
    if not model_file.is_file():
        return None, f"Kidney model file not found: {model_file}"
    if not config_file.is_file():
        return None, (
            f"Kidney preprocessing/class mapping configuration is missing: {config_file}. "
            "Add the verified kidney_model_config.json using the training notebook/dataset "
            "class_indices and the exact training preprocessing. No prediction was made."
        )
    try:
        import tensorflow as tf
    except ImportError:
        return None, "TensorFlow is required to load best_model.keras. Install tensorflow-cpu>=2.18,<2.22 from requirements.txt."

    try:
        with config_file.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        class_names = config.get("class_names")
        input_size = config.get("input_size")
        color_mode = config.get("color_mode")
        preprocessing = config.get("preprocessing")
        if not isinstance(class_names, list) or len(class_names) != 4 or set(class_names) != set(KIDNEY_CLASS_NAMES):
            return None, "Kidney config class_names must contain exactly Cyst, Normal, Stone, Tumor in verified training-index order."
        if not (isinstance(input_size, list) and len(input_size) == 2 and all(isinstance(v, int) and v > 0 for v in input_size)):
            return None, "Kidney config input_size must be [height, width] copied from the training setup."
        if color_mode not in ("rgb", "grayscale"):
            return None, "Kidney config color_mode must be 'rgb' or 'grayscale' as used during training."
        if not isinstance(preprocessing, dict) or preprocessing.get("type") not in ("rescale", "standardize", "none"):
            return None, "Kidney config preprocessing.type must be verified as rescale, standardize, or none."
        if preprocessing["type"] == "rescale" and not isinstance(preprocessing.get("scale"), (int, float)):
            return None, "For rescale preprocessing, set the verified numeric preprocessing.scale."
        if preprocessing["type"] == "standardize":
            if not isinstance(preprocessing.get("mean"), list) or not isinstance(preprocessing.get("std"), list):
                return None, "For standardize preprocessing, provide verified mean and std arrays."
        model = tf.keras.models.load_model(model_file, compile=False)
        output_shape = model.output_shape
        if isinstance(output_shape, list) or len(output_shape) != 2 or output_shape[-1] != 4:
            return None, f"Kidney model output shape {output_shape!r} is not a single four-class output. No prediction was made."
        input_shape = model.input_shape
        if isinstance(input_shape, list) or len(input_shape) != 4:
            return None, f"Unsupported kidney model input shape: {input_shape!r}"
        expected_channels = 3 if color_mode == "rgb" else 1
        if input_shape[-1] is not None and input_shape[-1] != expected_channels:
            return None, f"Config color_mode expects {expected_channels} channels but model input shape is {input_shape!r}."
        if input_shape[1] is not None and input_shape[1] != input_size[0]:
            return None, f"Configured input height {input_size[0]} does not match model input shape {input_shape!r}."
        if input_shape[2] is not None and input_shape[2] != input_size[1]:
            return None, f"Configured input width {input_size[1]} does not match model input shape {input_shape!r}."
        return {"model": model, "config": config, "class_names": class_names}, None
    except Exception as exc:
        return None, f"Unable to load the kidney Keras model/configuration: {type(exc).__name__}: {exc}"


def run_kidney_inference(loaded, img):
    """Return a validated four-class prediction and scores, not a segmentation mask."""
    import numpy as np

    model = loaded["model"]
    config = loaded["config"]
    names = loaded["class_names"]
    size = config["input_size"]
    mode = config["color_mode"]
    preprocessing = config["preprocessing"]
    image = img.convert("RGB" if mode == "rgb" else "L")
    image = image.resize((size[1], size[0]), Image.Resampling.BILINEAR)
    array = np.asarray(image, dtype=np.float32)
    if mode == "grayscale":
        array = array[..., None]
    kind = preprocessing["type"]
    if kind == "rescale":
        array = array * float(preprocessing["scale"])
    elif kind == "standardize":
        mean = np.asarray(preprocessing["mean"], dtype=np.float32)
        std = np.asarray(preprocessing["std"], dtype=np.float32)
        if np.any(std == 0):
            raise ValueError("Kidney preprocessing std values must not contain zero.")
        array = (array - mean) / std
    elif kind != "none":
        raise ValueError(f"Unsupported preprocessing type: {kind}")
    predictions = np.asarray(model.predict(np.expand_dims(array, axis=0), verbose=0))
    if predictions.shape != (1, 4):
        raise ValueError(f"Kidney model returned shape {predictions.shape}; expected (1, 4).")
    scores = predictions[0].astype(float)
    if not np.all(np.isfinite(scores)):
        raise ValueError("Kidney model returned non-finite class scores.")
    # Convert logits to a normalized score distribution only when output isn't already probabilities.
    if np.any(scores < 0) or not np.isclose(float(scores.sum()), 1.0, atol=1e-3):
        scores = np.exp(scores - np.max(scores))
        scores = scores / scores.sum()
    elif scores.sum() <= 0:
        raise ValueError("Kidney model returned invalid class scores.")
    index = int(np.argmax(scores))
    return names[index], float(scores[index] * 100), [float(v) for v in scores]


# ============================================================
# INFERENCE
# ============================================================
chest_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)


def run_chest_inference(model, class_names, img):
    tensor = chest_transform(img.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        if not isinstance(logits, torch.Tensor) or logits.ndim != 2:
            raise ValueError("Chest model output must have shape (batch, classes).")
        if logits.shape[0] != 1 or logits.shape[1] != len(class_names):
            raise ValueError(
                f"Chest model output shape {tuple(logits.shape)} does not match "
                f"one image and {len(class_names)} configured classes."
            )
        probs = torch.softmax(logits, dim=1)[0]
        if not torch.isfinite(probs).all():
            raise ValueError("Chest model returned non-finite class scores.")

    confidence, pred = torch.max(probs, 0)

    return (
        class_names[int(pred.item())],
        float(confidence.item() * 100),
        probs.cpu().tolist(),
    )


def status_card(
    test_name,
    value,
    unit,
    status,
    medical,
    simple,
    conclusion,
):
    abnormal = status.upper() in {"HIGH", "LOW"}
    css = "result-abnormal" if abnormal else "result-normal"

    badge = (
        "🔴 HIGH"
        if status.upper() == "HIGH"
        else "🔵 LOW"
        if status.upper() == "LOW"
        else "🟢 NORMAL"
    )

    st.markdown(
        f"""
        <div class="{css}">
            <div style="font-size:1.02rem;font-weight:800;color:#0f172a;">
                {test_name}
                <span style="font-weight:600;">— {value} {unit}</span>
                <span style="float:right;font-size:.78rem;">{badge}</span>
            </div>
            <div style="margin-top:9px;color:#334155;">
                <b>Medical overview:</b> {medical}
            </div>
            <div style="margin-top:6px;color:#475569;">
                {simple}
            </div>
            <div style="margin-top:9px;color:#1d4ed8;font-weight:700;">
                {conclusion}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:800;">
            🩺 MediVision
        </div>
        <div style="color:#8fa5bd;font-size:.83rem;margin:4px 0 18px;">
            Multimodal Health Suite
        </div>
        """,
        unsafe_allow_html=True,
    )

    module_choice = st.radio(
        "DIAGNOSTIC MODULE",
        (
            "🪨 Kidney Stone Detection",
            "🫁 Chest X-Ray Disease Detection",
            "📄 Medical Report Analyzer",
        ),
    )

    st.markdown("---")

    language = st.selectbox(
        "REPORT LANGUAGE / भाषा / भाषा",
        ["English", "हिंदी (Hindi)", "मराठी (Marathi)"],
        index=0,
    )

    lang_code = (
        "en"
        if language == "English"
        else ("hi" if "Hindi" in language else "mr")
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            padding:13px;
            border:1px solid rgba(255,255,255,.10);
            border-radius:14px;
            background:rgba(255,255,255,.035);
        ">
            <div style="
                font-size:.72rem;
                color:#7f95ad;
                text-transform:uppercase;
                letter-spacing:.08em;
            ">System status</div>
            <div style="margin-top:6px;font-weight:750;color:#a7f3d0;">
                ● Models ready on demand
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# MAIN
# ============================================================
hero()

# ============================================================
# MODULE 1 — KIDNEY
# ============================================================
if module_choice == "🪨 Kidney Stone Detection":
    st.markdown(
        '<div class="section-title">Kidney Stone Diagnostic Scanner</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Upload a renal CT or ultrasound image for educational '
        'segmentation-based screening.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title(
            "🖼️",
            "Imaging input",
            "Use a clear PNG, JPG or JPEG image.",
        )

        uploaded_file = st.file_uploader(
            "Choose CT / Ultrasound Image",
            type=["png", "jpg", "jpeg"],
            key="kidney_upload",
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")

            st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
            st.image(
                img,
                caption="Uploaded renal image",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown(
                f"""
                <div style="margin-top:12px;color:#64748b;font-size:.82rem;">
                    <b>File:</b> {uploaded_file.name}
                    &nbsp; • &nbsp;
                    <b>Resolution:</b> {img.width} × {img.height}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.info("Upload an image to enable the analysis workspace.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        panel_title(
            "🧠",
            "Four-class classification",
            "Cyst, Normal, Stone, and Tumor scores from the configured Keras model.",
        )

        if uploaded_file:
            if st.button(
                "🚀 Analyze Kidney Scan",
                use_container_width=True,
            ):
                with st.spinner("Loading the Keras model and checking its output…"):
                    model, err = load_kidney_classifier(
                        MODELS_DIR / "best_model.keras",
                        MODELS_DIR / "kidney_model_config.json",
                    )

                    if err:
                        st.error(err)
                    else:
                        try:
                            with st.spinner("Running four-class kidney classification…"):
                                result_text, confidence_val, probabilities = run_kidney_inference(model, img)
                            st.markdown(
                                f"""
                                <div class="result-info">
                                    <div style="font-size:1.08rem;font-weight:850;">
                                        Predicted class: {result_text}
                                    </div>
                                    <div style="margin-top:7px;color:#334155;">
                                        Highest model score: <b>{confidence_val:.2f}%</b>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            metric_cols = st.columns(4)
                            for idx, col in enumerate(metric_cols):
                                with col:
                                    st.metric(model["class_names"][idx], f"{probabilities[idx] * 100:.2f}%")
                            probability_df = pd.DataFrame({
                                "Class": model["class_names"],
                                "Model score (%)": [round(p * 100, 2) for p in probabilities],
                            })
                            st.markdown("#### 📊 Class score profile")
                            st.bar_chart(probability_df.set_index("Class"))
                            confidence_explanation(confidence_val, result_text, "Keras four-class kidney classifier")
                            st.markdown(
                                """
                                <div class="result-warning">
                                    <b>Medical safety note:</b> This is a model classification, not a confirmed diagnosis.
                                    The model score is not the clinical probability of disease. Review the scan with a qualified clinician.
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        except Exception as exc:
                            st.error(f"Kidney inference failed; no prediction was produced. Details: {type(exc).__name__}: {exc}")

        else:
            st.markdown(
                """
                <div class="result-info">
                    <b>Ready.</b><br>
                    Upload a scan on the left and the classification workflow
                    will appear here.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODULE 2 — CHEST X-RAY
# ============================================================
elif module_choice == "🫁 Chest X-Ray Disease Detection":
    st.markdown(
        '<div class="section-title">Chest X-Ray Diagnostic Analysis</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Three-class computer-vision screening using the supplied '
        'DenseNet121 checkpoint.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.05, 1], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        panel_title(
            "🩻",
            "Radiograph input",
            "Prefer a frontal PA/AP chest radiograph.",
        )

        uploaded_file = st.file_uploader(
            "Upload Chest X-Ray",
            type=["png", "jpg", "jpeg"],
            key="chest_upload",
        )

        if uploaded_file:
            img = Image.open(uploaded_file).convert("RGB")

            st.markdown('<div class="scan-frame">', unsafe_allow_html=True)
            st.image(
                img,
                caption="Uploaded radiograph",
                use_container_width=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("Upload a radiograph to enable screening.")

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        panel_title(
            "🧠",
            "AI screening",
            "DenseNet121 is restored from the checkpoint metadata.",
        )

        if uploaded_file:
            if st.button(
                "🚀 Run X-Ray Screening",
                use_container_width=True,
            ):
                with st.spinner(
                    "Extracting features and evaluating patterns…"
                ):
                    loaded, err = load_chest_model(
                        str(MODELS_DIR / "densenet121_3class_best.pth")
                    )

                    if err:
                        st.error(err)
                    else:
                        model, class_names = loaded

                        (
                            result_text,
                            confidence_val,
                            probabilities,
                        ) = run_chest_inference(
                            model,
                            class_names,
                            img,
                        )

                        st.markdown(
                            f"""
                            <div class="result-warning">
                                <div style="font-size:1.1rem;font-weight:850;">
                                    🟠 {result_text}
                                </div>
                                <div style="margin-top:7px;color:#475569;">
                                    Highest predicted class confidence:
                                    <b>{confidence_val:.2f}%</b>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        metric_cols = st.columns(len(class_names))

                        for idx, col in enumerate(metric_cols):
                            with col:
                                label = class_names[idx]
                                prob = (
                                    probabilities[idx] * 100
                                    if idx < len(probabilities)
                                    else 0
                                )
                                st.metric(
                                    label,
                                    f"{prob:.2f}%",
                                )

                        st.progress(
                            min(max(int(confidence_val), 0), 100)
                        )

                        confidence_explanation(
                            confidence_val,
                            result_text,
                            "DenseNet121 chest X-ray classifier",
                        )

                        st.markdown("#### 📊 Class probability profile")

                        probability_df = pd.DataFrame(
                            {
                                "Class": class_names,
                                "Probability (%)": [
                                    round(p * 100, 2)
                                    for p in probabilities
                                ],
                            }
                        )

                        st.bar_chart(
                            probability_df.set_index("Class")
                        )

                        st.markdown(
                            """
                            <div class="result-info">
                                📋 <b>Clinical correlation:</b>
                                A radiograph model output should be reviewed
                                with the original image, patient history,
                                examination and qualified radiologist
                                interpretation.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

        else:
            st.markdown(
                """
                <div class="result-info">
                    <b>Ready.</b><br>
                    Upload a chest X-ray on the left to run the model.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# MODULE 3 — MEDICAL REPORT
# ============================================================
else:
    st.markdown(
        '<div class="section-title">Automated Lab Report Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-subtitle">'
        'Extract common laboratory parameters, compare them with configured '
        'reference ranges, and explain results in English, Hindi or Marathi.'
        '</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([.9, 1.6], gap="large")

    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        panel_title(
            "📄",
            "Report input",
            "Supported: PDF, PNG, JPG, JPEG and TXT.",
        )

        uploaded_file = st.file_uploader(
            "Upload Lab Document",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            key="report_upload",
        )

        if uploaded_file:
            st.success(f"Loaded: {uploaded_file.name}")
            st.caption(
                f"Size: {uploaded_file.size / 1024:.1f} KB"
            )
        else:
            st.info("Upload a report to start extraction.")

        st.markdown(
            """
            <div class="result-info">
                🌐 <b>Language:</b> selected from the sidebar.<br>
                🔎 <b>Engine:</b> rule-based extraction + configured reference ranges.<br>
                🛡️ <b>Privacy:</b> temporary uploaded files are removed after processing.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)

        panel_title(
            "🔬",
            "Report analysis",
            "Extract → classify → explain → export.",
        )

        if uploaded_file:
            if st.button(
                "🔍 Extract & Analyze Lab Report",
                use_container_width=True,
            ):
                file_ext = Path(uploaded_file.name).suffix.lower()
                tmp_path = None

                try:
                    with st.spinner(
                        "Extracting report text and applying the rules engine…"
                    ):
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            suffix=file_ext,
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        analysis_results = parse_and_analyze_report(
                            tmp_path,
                            lang=lang_code,
                        )

                        raw_text = extract_text_from_file(tmp_path)
                        extracted_vals = extract_values(raw_text)
                        findings = detect_findings(extracted_vals)

                    if not analysis_results:
                        st.warning(
                            "No supported lab parameters were recognized. "
                            "Check report quality, OCR availability, or "
                            "standard test labels."
                        )

                    else:
                        abnormal_count = sum(
                            1
                            for item in analysis_results
                            if item["Status"].upper()
                            in {"LOW", "HIGH"}
                        )

                        normal_count = (
                            len(analysis_results) - abnormal_count
                        )

                        c1, c2, c3 = st.columns(3)

                        with c1:
                            st.markdown(
                                f"""
                                <div class="metric">
                                    <div class="metric-label">Parameters</div>
                                    <div class="metric-value">
                                        {len(analysis_results)}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with c2:
                            st.markdown(
                                f"""
                                <div class="metric">
                                    <div class="metric-label">Within range</div>
                                    <div class="metric-value">
                                        {normal_count}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with c3:
                            st.markdown(
                                f"""
                                <div class="metric">
                                    <div class="metric-label">Outside range</div>
                                    <div class="metric-value">
                                        {abnormal_count}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.markdown("#### 📊 Extracted Lab Summary")

                        df = pd.DataFrame(analysis_results)

                        preferred_columns = [
                            "Test",
                            "Value",
                            "Unit",
                            "Reference Range",
                            "Status",
                        ]

                        available_columns = [
                            c for c in preferred_columns
                            if c in df.columns
                        ]

                        df_display = df[available_columns]

                        st.dataframe(
                            df_display,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.markdown("#### 💡 Simplified explanations")

                        for item in analysis_results:
                            status_card(
                                item["Test"],
                                item["Value"],
                                item["Unit"],
                                item["Status"],
                                item["MedicalExplanation"],
                                item["SimpleExplanation"],
                                item["Conclusion"],
                            )

                        if findings:
                            st.markdown(
                                "#### 🔎 Rule-based clinical patterns"
                            )

                            for finding in findings:
                                st.markdown(
                                    f"""
                                    <div class="result-warning">
                                        <b>{finding['Condition']}</b><br>
                                        <span style="color:#475569;">
                                            {finding['Reason']}
                                        </span><br>
                                        <span style="color:#334155;">
                                            {finding['Interpretation']}
                                        </span>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        st.markdown(
                            f"""
                            <div class="result-info">
                                📝 <b>Overall summary:</b>
                                {len(analysis_results)} parameter(s) were
                                extracted and {abnormal_count} fell outside
                                the configured reference limits.
                                Please discuss abnormal results with a
                                qualified healthcare professional.
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                        csv_data = df_display.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            "⬇️ Download extracted results (CSV)",
                            data=csv_data,
                            file_name="medivision_lab_results.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                except Exception as exc:
                    st.error(
                        f"Error processing report: {exc}"
                    )

                finally:
                    if tmp_path and os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

        else:
            st.markdown(
                """
                <div class="result-info">
                    <b>Ready for a report.</b><br>
                    Choose a PDF, image or TXT laboratory report to begin.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# GLOBAL DISCLAIMER
# ============================================================
st.markdown(
    """
    <div class="result-warning" style="margin-top:28px;">
        ⚠️ <b>Mandatory educational disclaimer:</b>
        MediVision is an educational/prototype application. Its
        computer-vision outputs, extracted values and rule-based
        explanations are not a substitute for professional medical
        diagnosis, treatment or radiologist/physician review.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer-note">'
    'MediVision • Multimodal Health Suite • Educational Prototype'
    '</div>',
    unsafe_allow_html=True,
)
