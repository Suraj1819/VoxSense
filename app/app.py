# app/app.py
import sys
import asyncio
from pathlib import Path

# Fix Windows asyncio ProactorEventLoop error
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import torch
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent.parent
sys.path.append(str(ROOT_DIR))

from src.models.cnn_model import AudioCNN

# Page configuration
st.set_page_config(
    page_title="VoxSense",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * { font-family: 'Inter', sans-serif; }

    .stApp { background: #0a0a0f; }

    .main-header {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
        margin-bottom: 1rem;
    }

    .header-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #ffd200 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }

    .header-sub {
        color: #a0a0b8;
        font-size: 1rem;
        font-weight: 400;
        margin-top: 0.4rem;
        letter-spacing: 0.5px;
    }

    .emotion-card {
        padding: 2.5rem 2rem;
        border-radius: 24px;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin: 1rem 0;
    }

    .emotion-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        pointer-events: none;
    }

    .emotion-angry {
        background: linear-gradient(135deg, #ff4757, #ff6348, #ff7f50);
        box-shadow: 0 25px 60px rgba(255, 71, 87, 0.35);
    }
    .emotion-happy {
        background: linear-gradient(135deg, #ffa502, #ffd32a, #fff200);
        box-shadow: 0 25px 60px rgba(255, 165, 2, 0.35);
    }
    .emotion-normal {
        background: linear-gradient(135deg, #2ed573, #7bed9f, #a3f7bf);
        box-shadow: 0 25px 60px rgba(46, 213, 115, 0.35);
    }
    .emotion-sad {
        background: linear-gradient(135deg, #74b9ff, #0984e3, #0652DD);
        box-shadow: 0 25px 60px rgba(9, 132, 227, 0.35);
    }
    .emotion-fearful {
        background: linear-gradient(135deg, #a29bfe, #6c5ce7, #4834d4);
        box-shadow: 0 25px 60px rgba(108, 92, 231, 0.35);
    }
    .emotion-calm {
        background: linear-gradient(135deg, #55efc4, #00b894, #009432);
        box-shadow: 0 25px 60px rgba(0, 184, 148, 0.35);
    }
    .emotion-excited {
        background: linear-gradient(135deg, #fd79a8, #e84393, #c44569);
        box-shadow: 0 25px 60px rgba(232, 67, 147, 0.35);
    }
    .emotion-anxious {
        background: linear-gradient(135deg, #ffeaa7, #fdcb6e, #f6b93b);
        box-shadow: 0 25px 60px rgba(253, 203, 110, 0.35);
    }
    .emotion-aggressive {
        background: linear-gradient(135deg, #ff7675, #d63031, #b71540);
        box-shadow: 0 25px 60px rgba(214, 48, 49, 0.35);
    }
    .emotion-default {
        background: linear-gradient(135deg, #dfe6e9, #b2bec3, #636e72);
        box-shadow: 0 25px 60px rgba(178, 190, 195, 0.25);
    }

    .emotion-icon { font-size: 4.5rem; margin-bottom: 0.5rem; filter: drop-shadow(0 4px 8px rgba(0,0,0,0.2)); }
    .emotion-label { font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: #fff; text-shadow: 0 2px 10px rgba(0,0,0,0.2); }
    .emotion-desc { font-size: 0.95rem; opacity: 0.95; margin-bottom: 1rem; color: #fff; font-weight: 500; }
    .emotion-confidence { font-size: 2.8rem; font-weight: 900; color: #fff; text-shadow: 0 2px 15px rgba(0,0,0,0.2); }

    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(20px);
        margin-bottom: 1rem;
    }

    .metric-mini {
        background: rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.2s;
    }

    .metric-mini:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.15);
    }

    .metric-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #ffffff;
    }

    .metric-label {
        font-size: 0.7rem;
        color: #b0b0c8;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.4rem;
        font-weight: 600;
    }

    .section-title {
        font-size: 1rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
        letter-spacing: 0.3px;
    }

    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #555568;
        font-size: 0.85rem;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .upload-section {
        border: 2px dashed rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        background: rgba(255,255,255,0.02);
        transition: all 0.3s;
        margin-bottom: 1rem;
    }

    .upload-section:hover {
        border-color: rgba(240, 147, 251, 0.5);
        background: rgba(240, 147, 251, 0.05);
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stDeployButton { visibility: hidden; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.06);
        border-radius: 12px;
        color: #b0b0c8;
        font-size: 0.85rem;
        padding: 0.5rem 1.2rem;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #f093fb, #f5576c) !important;
        color: #fff !important;
    }

    .stProgress > div > div > div {
        background: linear-gradient(90deg, #f093fb, #f5576c, #ffd200);
        border-radius: 10px;
    }

    .stFileUploader > div > div > div {
        color: #ffffff !important;
    }

    .stFileUploader label {
        color: #b0b0c8 !important;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: #fff;
        font-weight: 700;
        border: none;
        border-radius: 14px;
        padding: 0.7rem 2rem;
        font-size: 0.95rem;
        letter-spacing: 0.3px;
        transition: all 0.2s;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 30px rgba(245, 87, 108, 0.4);
    }

    .stExpander > div > div {
        background: rgba(255,255,255,0.04);
        border-radius: 14px;
        color: #ffffff;
    }

    .stExpander > div > div > div > p {
        color: #e0e0f0;
    }

    .stCaption, .stCaption p {
        color: #b0b0c8 !important;
    }

    .stAlert {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 14px !important;
        color: #ffffff !important;
    }

    .stSpinner > div {
        border-top-color: #f093fb !important;
    }

    audio {
        width: 100%;
        border-radius: 14px;
        outline: none;
    }

    .stDownloadButton > button {
        background: rgba(255,255,255,0.08);
        color: #ffffff;
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stDownloadButton > button:hover {
        background: rgba(255,255,255,0.14);
        border-color: rgba(240, 147, 251, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ── Emotion config (animal-agnostic) ──────────────────────────
EMOTION_ICONS = {
    'angry': '😠', 'happy': '😊', 'normal': '😐',
    'sad': '😢', 'fearful': '😨', 'calm': '😌',
    'excited': '🤩', 'anxious': '😰', 'aggressive': '🔥',
    'distressed': '😿', 'playful': '😻', 'alert': '🦉',
}
EMOTION_DESCS = {
    'angry': 'Aggressive / Threatening',
    'happy': 'Joyful / Content',
    'normal': 'Calm / Neutral',
    'sad': 'Distressed / Low mood',
    'fearful': 'Scared / Anxious',
    'calm': 'Relaxed / Peaceful',
    'excited': 'Highly Energetic / Aroused',
    'anxious': 'Nervous / Uneasy',
    'aggressive': 'Hostile / Defensive',
    'distressed': 'In distress / Seeking help',
    'playful': 'Playful / Friendly',
    'alert': 'Alert / Attentive',
}
EMOTION_RECS = {
    'angry': ("🟠 Warning sign detected. Give the animal space. Identify and remove potential stressors. Do NOT approach or make sudden movements.", "warning"),
    'happy': ("🟢 Positive emotional state. The animal seems comfortable and safe. Continue current interaction if appropriate.", "success"),
    'normal': ("🔵 Baseline neutral state. No immediate concerns. Continue monitoring for any behavioral changes.", "info"),
    'sad': ("💙 Possible distress or discomfort. Check for illness, injury, or environmental changes. Consult a vet if persistent.", "info"),
    'fearful': ("🟣 Fear response detected. Remove perceived threats. Speak softly and allow the animal to retreat to a safe space.", "warning"),
    'calm': ("💚 Relaxed state. The animal feels safe in its environment. Ideal baseline for well-being assessment.", "success"),
    'excited': ("🩷 High arousal state. May be positive (play) or overstimulation. Monitor for escalation into anxiety or aggression.", "info"),
    'anxious': ("🟡 Anxiety detected. Look for environmental triggers. Provide reassurance and a quiet, familiar space.", "warning"),
    'aggressive': ("🔴 High aggression risk. Do NOT approach. Secure the area and seek professional help if needed.", "error"),
    'distressed': ("💔 Animal may be in pain or danger. Inspect for injuries, illness, or entrapment. Seek veterinary attention.", "error"),
    'playful': ("💜 Playful mood detected. Great sign of well-being! Engage if safe and appropriate.", "success"),
    'alert': ("🟠 Attentive state. Animal is focused on something in the environment. Check for potential triggers.", "info"),
}
BAR_COLORS = {
    'angry': 'linear-gradient(90deg, #ff4757, #ff6348)',
    'happy': 'linear-gradient(90deg, #ffa502, #ffd32a)',
    'normal': 'linear-gradient(90deg, #2ed573, #7bed9f)',
    'sad': 'linear-gradient(90deg, #74b9ff, #0984e3)',
    'fearful': 'linear-gradient(90deg, #a29bfe, #6c5ce7)',
    'calm': 'linear-gradient(90deg, #55efc4, #00b894)',
    'excited': 'linear-gradient(90deg, #fd79a8, #e84393)',
    'anxious': 'linear-gradient(90deg, #ffeaa7, #fdcb6e)',
    'aggressive': 'linear-gradient(90deg, #ff7675, #d63031)',
    'distressed': 'linear-gradient(90deg, #81ecec, #00cec9)',
    'playful': 'linear-gradient(90deg, #fd79a8, #e84393)',
    'alert': 'linear-gradient(90deg, #fab1a0, #e17055)',
}


def _clean_class(raw: str) -> str:
    """Strip common prefixes like 'dog_', 'cat_', 'bird_' etc."""
    for prefix in ('dog_', 'cat_', 'bird_', 'cow_', 'horse_', 'lion_', 'elephant_', 'monkey_'):
        if raw.lower().startswith(prefix):
            return raw[len(prefix):]
    return raw


# ── Model loader ──────────────────────────────────────────────
@st.cache_resource
def load_model(model_path):
    try:
        checkpoint = torch.load(model_path, weights_only=False, map_location='cpu')
        num_classes = checkpoint.get('num_classes', 3)
        model = AudioCNN(num_classes=num_classes)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        le = checkpoint['label_encoder']
        if not hasattr(le, 'classes_') or len(le.classes_) == 0:
            raise ValueError("Invalid label encoder")
        metrics = checkpoint.get('metrics', {})
        return model, le, metrics
    except Exception as e:
        st.error(f"Model load failed: {str(e)}")
        return None, None, None


# ── Audio processing ──────────────────────────────────────────
def create_spectrogram(audio_path, save_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000, hop_length=512, n_fft=2048)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        img = librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', cmap='magma', ax=ax)

        # Colorbar — matplotlib needs RGBA tuple, NOT CSS rgba string
        cbar = plt.colorbar(img, ax=ax, format='%+2.0f dB')
        cbar.ax.yaxis.set_tick_params(color='#ffffff')
        cbar.outline.set_edgecolor((1.0, 1.0, 1.0, 0.15))
        plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#ffffff', fontsize=9)

        ax.set_title('Mel Spectrogram', color='#ffffff', fontsize=14, fontweight=700, pad=12)
        ax.set_xlabel('Time (s)', color='#ffffff', fontsize=11, fontweight=500, labelpad=8)
        ax.set_ylabel('Frequency (Hz)', color='#ffffff', fontsize=11, fontweight=500, labelpad=8)
        ax.tick_params(colors='#ffffff', which='both')
        for label in ax.get_xticklabels():
            label.set_color('#ffffff'); label.set_fontsize(9)
        for label in ax.get_yticklabels():
            label.set_color('#ffffff'); label.set_fontsize(9)

        # Spine colors — matplotlib needs RGBA tuple, NOT CSS rgba string
        for spine in ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.15))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
        plt.close()
        return True, duration, y, sr
    except Exception as e:
        st.error(f"Spectrogram failed: {str(e)}")
        return False, 0, None, None


def create_waveform_plot(y, sr, save_path):
    try:
        fig, ax = plt.subplots(figsize=(10, 3), facecolor='#0a0a0f')
        ax.set_facecolor('#0a0a0f')
        time = np.linspace(0, len(y) / sr, len(y))

        ax.plot(time, y, color='#f093fb', linewidth=0.8, alpha=0.95)
        ax.fill_between(time, y, alpha=0.15, color='#f093fb')

        ax.set_title('Waveform', color='#ffffff', fontsize=14, fontweight=700, pad=12)
        ax.set_xlabel('Time (s)', color='#ffffff', fontsize=11, fontweight=500, labelpad=8)
        ax.set_ylabel('Amplitude', color='#ffffff', fontsize=11, fontweight=500, labelpad=8)
        ax.tick_params(colors='#ffffff', which='both')
        for label in ax.get_xticklabels():
            label.set_color('#ffffff'); label.set_fontsize(9)
        for label in ax.get_yticklabels():
            label.set_color('#ffffff'); label.set_fontsize(9)
        ax.grid(True, alpha=0.08, color='#ffffff')
        ax.set_xlim([0, len(y)/sr])

        # Spine colors — matplotlib needs RGBA tuple, NOT CSS rgba string
        for spine in ax.spines.values():
            spine.set_edgecolor((1.0, 1.0, 1.0, 0.15))

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0a0a0f')
        plt.close()
        return True
    except:
        return False


def extract_features(y, sr):
    features = {
        'duration': len(y) / sr, 'rms_energy': 0.0, 'peak_amplitude': 0.0,
        'zero_crossing_rate': 0.0, 'spectral_centroid': 0.0,
        'pitch_mean': 0.0, 'tempo': 0.0
    }
    try:
        features['rms_energy'] = float(np.sqrt(np.mean(y**2)))
        features['peak_amplitude'] = float(np.max(np.abs(y)))
        try:
            features['zero_crossing_rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)[0]))
        except: pass
        try:
            features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)[0]))
        except: pass
        try:
            pitches, mags = librosa.piptrack(y=y, sr=sr)
            if mags.size > 0:
                pv = pitches[mags > np.max(mags) * 0.1]
                if len(pv) > 0: features['pitch_mean'] = float(np.mean(pv))
        except: pass
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray): tempo = tempo[0] if len(tempo) > 0 else 0
            features['tempo'] = float(tempo) if tempo and tempo > 0 else 0.0
        except: pass
    except: pass
    return features


# ── Prediction ────────────────────────────────────────────────
def predict_emotion(model, img_tensor, label_encoder):
    try:
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            idx = torch.argmax(output, dim=1).item()

            num_classes = len(label_encoder.classes_)
            if idx >= num_classes:
                return "unknown", 0.0, {}, 0.0

            result = label_encoder.inverse_transform([idx])
            if len(result) == 0:
                return "unknown", 0.0, {}, 0.0

            pred_class = result[0]
            confidence = probs[idx].item() * 100
            class_probs = {str(c): float(p) * 100 for c, p in zip(label_encoder.classes_, probs)}
            entropy = -torch.sum(probs * torch.log(probs + 1e-10)).item()
            return pred_class, confidence, class_probs, entropy
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return "unknown", 0.0, {}, 0.0


def prepare_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    img = Image.open(image_path).convert('L')
    return transform(img).unsqueeze(0)


def format_val(v, fmt="{:.1f}", default="—"):
    if v is None: return default
    try: return fmt.format(float(v))
    except: return str(v) if v else default


# ── Main ──────────────────────────────────────────────────────
def main():
    st.markdown("""
    <div class="main-header">
        <p class="header-title">🐾 VoxSense</p>
        <p class="header-sub">Animal Emotion Recognition from Vocalizations</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop your audio file here",
        type=["wav"],
        label_visibility="collapsed",
        help="Upload a .wav file of any animal's vocalization"
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if 'analyze_clicked' not in st.session_state:
        st.session_state.analyze_clicked = False

    if uploaded_file:
        col_analyze, col_preview = st.columns([1, 3])
        with col_analyze:
            if st.button("⬆ Analyze", use_container_width=True, type="primary"):
                st.session_state.analyze_clicked = True
        with col_preview:
            st.audio(uploaded_file, format='audio/wav')

    if uploaded_file and st.session_state.analyze_clicked:
        temp_audio = Path("temp_audio.wav")
        temp_spec = Path("temp_spec.png")
        temp_wave = Path("temp_wave.png")

        try:
            with open(temp_audio, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Analyzing audio..."):
                success, duration, y, sr = create_spectrogram(temp_audio, temp_spec)
                if not success or y is None:
                    st.error("Failed to process audio")
                    st.stop()
                create_waveform_plot(y, sr, temp_wave)
                features = extract_features(y, sr)

            # Load Model
            model_path = ROOT_DIR / "models" / "best_cnn_model.pth"
            if not model_path.exists():
                st.error(f"Model not found: {model_path}")
                st.stop()

            model, le, metrics = load_model(model_path)
            if model is None:
                st.stop()

            with st.spinner("Predicting emotion..."):
                img_tensor = prepare_image(temp_spec)
                pred_class, confidence, class_probs, entropy = predict_emotion(model, img_tensor, le)

            if pred_class == "unknown":
                st.error("Prediction failed")
                st.stop()

            # Clean class name
            class_key = _clean_class(pred_class).lower()
            emotion_name = class_key.upper()

            icon = EMOTION_ICONS.get(class_key, '🐾')
            desc = EMOTION_DESCS.get(class_key, 'Unknown emotional state')
            card_class = f"emotion-{class_key}" if f"emotion-{class_key}" in (
                'emotion-angry', 'emotion-happy', 'emotion-normal', 'emotion-sad',
                'emotion-fearful', 'emotion-calm', 'emotion-excited', 'emotion-anxious',
                'emotion-aggressive', 'emotion-distressed', 'emotion-playful', 'emotion-alert'
            ) else "emotion-default"

            # ── Result Card ───────────────────────────────────
            st.markdown(f"""
            <div class="emotion-card {card_class}">
                <div class="emotion-icon">{icon}</div>
                <div class="emotion-label">{emotion_name}</div>
                <div class="emotion-desc">{desc}</div>
                <div class="emotion-confidence">{confidence:.1f}%</div>
                <div style="font-size:0.8rem;opacity:0.8;margin-top:0.3rem;color:#fff;font-weight:500">confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Probability Bars ──────────────────────────────
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability Distribution</div>', unsafe_allow_html=True)

            sorted_probs = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)

            for cls, prob in sorted_probs:
                label = _clean_class(cls).title()
                clean_key = _clean_class(cls).lower()
                bar_bg = BAR_COLORS.get(clean_key, 'linear-gradient(90deg, #f093fb, #f5576c)')
                st.markdown(f"""
                <div style="margin-bottom:1rem">
                    <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                        <span style="color:#e0e0f0;font-size:0.9rem;font-weight:600">{label}</span>
                        <span style="color:#ffffff;font-weight:700;font-size:0.9rem">{prob:.1f}%</span>
                    </div>
                    <div style="background:rgba(255,255,255,0.06);border-radius:12px;height:10px;overflow:hidden">
                        <div style="width:{prob}%;height:100%;background:{bar_bg};border-radius:12px;transition:width 0.5s"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Visualizations + Features ─────────────────────
            col_viz, col_feat = st.columns(2)

            with col_viz:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Visualizations</div>', unsafe_allow_html=True)
                tab1, tab2 = st.tabs(["🎵 Spectrogram", "〰️ Waveform"])
                with tab1:
                    if temp_spec.exists():
                        st.image(str(temp_spec), use_column_width=True)
                with tab2:
                    if temp_wave.exists():
                        st.image(str(temp_wave), use_column_width=True)
                st.markdown(f'<p style="color:#b0b0c8;font-size:0.85rem;margin-top:0.5rem">{duration:.2f}s &bull; {sr}Hz</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            with col_feat:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Audio Features</div>', unsafe_allow_html=True)

                feat_items = [
                    ("⏱ Duration", f"{features.get('duration', 0):.2f}s"),
                    ("⚡ RMS Energy", f"{features.get('rms_energy', 0):.4f}"),
                    ("📈 Peak Amp", f"{features.get('peak_amplitude', 0):.3f}"),
                    ("🔀 ZCR", f"{features.get('zero_crossing_rate', 0):.4f}"),
                    ("🔊 Spec Centroid", f"{features.get('spectral_centroid', 0):.0f}Hz"),
                    ("🎶 Pitch Mean", f"{features.get('pitch_mean', 0):.0f}Hz"),
                    ("🥁 Tempo", f"{features.get('tempo', 0):.0f} BPM" if features.get('tempo', 0) > 0 else "— BPM"),
                ]

                for i in range(0, len(feat_items), 2):
                    cols = st.columns(2)
                    for j, col in enumerate(cols):
                        if i + j < len(feat_items):
                            label, value = feat_items[i + j]
                            col.markdown(f"""
                            <div class="metric-mini">
                                <div class="metric-value">{value}</div>
                                <div class="metric-label">{label}</div>
                            </div>
                            """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)

            # ── Model Metrics ─────────────────────────────────
            if metrics:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown('<div class="section-title">Model Performance</div>', unsafe_allow_html=True)
                m_cols = st.columns(4)
                m_items = [
                    ("🎯 Accuracy", metrics.get('accuracy'), "{:.1f}%"),
                    ("⚖️ F1 Score", metrics.get('f1_score'), "{:.3f}"),
                    ("📌 Precision", metrics.get('precision'), "{:.3f}"),
                    ("🔍 Recall", metrics.get('recall'), "{:.3f}"),
                ]
                for col, (name, val, fmt) in zip(m_cols, m_items):
                    col.markdown(f"""
                    <div class="metric-mini">
                        <div class="metric-value">{format_val(val, fmt)}</div>
                        <div class="metric-label">{name}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Recommendation ────────────────────────────────
            rec_text, rec_type = EMOTION_RECS.get(
                class_key,
                ("🐾 Emotion detected. Monitor the animal's behavior for context and consult a professional if needed.", "info")
            )
            getattr(st, rec_type)(rec_text)

            # ── Export ────────────────────────────────────────
            with st.expander("📄 Export Report"):
                report = f"""VoxSense — Animal Emotion Analysis Report
{'='*50}
Date      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
File      : {uploaded_file.name}
Duration  : {duration:.2f}s | Sample Rate: {sr}Hz

Prediction: {emotion_name} ({confidence:.1f}%)
Entropy   : {entropy:.3f}

Probabilities:
{chr(10).join(f"  {k.replace('dog_','').replace('cat_','').replace('bird_','').title()}: {v:.1f}%" for k,v in sorted_probs)}

Audio Features:
  Duration         = {features.get('duration',0):.2f}s
  RMS Energy       = {features.get('rms_energy',0):.4f}
  Peak Amplitude   = {features.get('peak_amplitude',0):.3f}
  Zero Crossing    = {features.get('zero_crossing_rate',0):.4f}
  Spectral Centroid= {features.get('spectral_centroid',0):.0f}Hz
  Pitch Mean       = {features.get('pitch_mean',0):.0f}Hz
  Tempo            = {features.get('tempo',0):.0f} BPM

Model Metrics:
  Accuracy  = {format_val(metrics.get('accuracy'),'{:.1f}%')}
  F1 Score  = {format_val(metrics.get('f1_score'),'{:.3f}')}
  Precision = {format_val(metrics.get('precision'),'{:.3f}')}
  Recall    = {format_val(metrics.get('recall'),'{:.3f}')}

{'='*50}
VoxSense by • Neural Hashira • © 2026
"""
                st.download_button(
                    "⬇ Download Report",
                    report,
                    f"voxsense_{emotion_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    "text/plain"
                )

        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            for f in [temp_audio, temp_spec, temp_wave]:
                try:
                    if f.exists(): f.unlink()
                except: pass

    elif not uploaded_file:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem">
            <p style="font-size:4rem;margin-bottom:1.5rem">🐾</p>
            <p style="font-size:1.2rem;color:#e0e0f0;font-weight:600;margin-bottom:0.5rem">No audio uploaded yet</p>
            <p style="font-size:0.95rem;color:#707088;font-weight:400">
                Upload a WAV file to analyze any animal's vocalization emotion<br>
                <span style="font-size:0.8rem;color:#555568">Supports dogs, cats, birds, cattle, and more</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class="footer">
        VoxSense • Neural Hashira • Suraj Kumar • © 2026
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()