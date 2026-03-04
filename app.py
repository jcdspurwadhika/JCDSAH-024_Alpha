"""
app.py – Platform Prediksi Churn Pelanggan E-Commerce
Aplikasi Streamlit produksi untuk prediksi churn pelanggan e-commerce
Model: LightGBM | Dataset: E-Commerce India (5.630 pelanggan)
"""

import pickle
import warnings
import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.metrics import (
    roc_curve, precision_recall_curve, confusion_matrix,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {background: #0F172A;}
[data-testid="stSidebar"] {background: #1E293B; border-right: 1px solid #334155;}
[data-testid="stSidebar"] * {color: #E2E8F0 !important;}
h1,h2,h3,h4,h5,h6 {color: #F1F5F9 !important;}
p, li, span, label, div {color: #CBD5E1;}
.stSelectbox label, .stSlider label, .stNumberInput label {color: #94A3B8 !important; font-size: 0.82rem !important;}

/* ── Metric Cards ── */
.kpi-card {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155; border-radius: 12px;
    padding: 20px 24px; text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.kpi-value {font-size: 2.2rem; font-weight: 800; line-height: 1.1;}
.kpi-label {font-size: 0.80rem; color: #64748B; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px;}
.kpi-delta {font-size: 0.78rem; margin-top: 4px;}

/* ── Section header ── */
.section-header {
    border-left: 4px solid #3B82F6;
    padding-left: 12px; margin: 24px 0 16px 0;
    font-size: 1.15rem; font-weight: 700; color: #F1F5F9 !important;
}

/* ── Risk Badges ── */
.risk-high   {background:#7F1D1D; color:#FECACA; border:1px solid #EF4444; padding:6px 16px; border-radius:999px; font-weight:700; font-size:1rem;}
.risk-medium {background:#78350F; color:#FDE68A; border:1px solid #F59E0B; padding:6px 16px; border-radius:999px; font-weight:700; font-size:1rem;}
.risk-low    {background:#052E16; color:#BBF7D0; border:1px solid #10B981; padding:6px 16px; border-radius:999px; font-weight:700; font-size:1rem;}

/* ── Info / Alert Boxes ── */
.alert-box {
    border-radius: 10px; padding: 14px 18px; margin: 12px 0;
    border-left: 4px solid;
}
.alert-danger  {background:#1C0A0A; border-color:#EF4444; color:#FCA5A5;}
.alert-warning {background:#1C1000; border-color:#F59E0B; color:#FDE68A;}
.alert-success {background:#021C0E; border-color:#10B981; color:#6EE7B7;}
.alert-info    {background:#0D1F3C; border-color:#3B82F6; color:#93C5FD;}

/* ── Table ── */
.stDataFrame {border-radius: 8px; overflow: hidden;}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #1D4ED8);
    color: white; border: none; border-radius: 8px;
    font-weight: 600; padding: 10px 24px; width: 100%;
    box-shadow: 0 2px 12px rgba(37,99,235,0.35);
    transition: all 0.2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF);
    box-shadow: 0 4px 20px rgba(37,99,235,0.55);
    transform: translateY(-1px);
}

/* ── Sidebar branding ── */
.brand-header {text-align:center; padding: 20px 0 24px 0; border-bottom: 1px solid #334155; margin-bottom: 18px;}
.brand-name   {font-size:1.45rem; font-weight:800; color:#F1F5F9 !important;}
.brand-tag    {font-size:0.72rem; color:#64748B !important; letter-spacing:1.5px; text-transform:uppercase;}

/* ── Sidebar nav section label ── */
.nav-section-title {
    font-size: 0.65rem; color: #475569 !important; letter-spacing: 2px;
    text-transform: uppercase; font-weight: 700;
    margin: 0 0 8px 4px; padding: 0;
}
/* Remove gap between radio options */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 0 !important;
}
/* Style each radio label wrapper */
[data-testid="stSidebar"] div[role="radiogroup"] label {
    width: 100% !important;
    border-radius: 10px !important;
    margin-bottom: 3px !important;
    transition: background 0.18s, border-color 0.18s, box-shadow 0.18s !important;
}
/* Style the baseweb radio container */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] {
    display: flex !important;
    align-items: center !important;
    padding: 10px 14px !important;
    border-radius: 10px !important;
    border: 1px solid transparent !important;
    cursor: pointer !important;
    transition: background 0.18s, border-color 0.18s, box-shadow 0.18s !important;
    background: transparent !important;
    width: 100% !important;
}
/* Hide the radio circle */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}
/* Style the text — default (inactive) */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] > div:last-child {
    color: #64748B !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0 !important;
    letter-spacing: 0.01em !important;
    transition: color 0.18s !important;
}
/* ── Hover ── */
[data-testid="stSidebar"] div[role="radiogroup"] label:hover [data-baseweb="radio"] {
    background: rgba(59,130,246,0.12) !important;
    border-color: rgba(59,130,246,0.35) !important;
    box-shadow: 0 0 0 0 transparent !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover [data-baseweb="radio"] > div:last-child {
    color: #93C5FD !important;
    font-weight: 600 !important;
}
/* ── Active / checked — uses :has() for reliable detection ── */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"]:has(input[type="radio"]:checked) {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    border-color: #60A5FA !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.55), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    border-left: 3px solid #93C5FD !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"]:has(input[type="radio"]:checked) > div:last-child {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    text-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
}
/* Remove focus ring */
[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"]:focus-within {
    outline: none !important;
}
/* Fallback — input:checked sibling approach */
[data-testid="stSidebar"] div[role="radiogroup"] input[type="radio"]:checked ~ div,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) [data-baseweb="radio"] {
    background: linear-gradient(135deg, #1D4ED8 0%, #2563EB 100%) !important;
    border-color: #60A5FA !important;
    border-left: 3px solid #93C5FD !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.55) !important;
}

/* ── Divider ── */
.fancy-divider {border:none; border-top:1px solid #1E293B; margin: 20px 0;}

/* ── Prediction result panel ── */
.pred-panel {
    border-radius: 14px; padding: 28px 32px;
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    text-align: center;
}
.pred-churn    {background: linear-gradient(135deg,#450A0A,#1C0A0A); border: 1px solid #EF4444;}
.pred-nochurn  {background: linear-gradient(135deg,#052E16,#021C0E); border: 1px solid #10B981;}
.pred-score    {font-size:3.2rem; font-weight:900; line-height:1;}
.pred-subtitle {font-size:0.88rem; color:#94A3B8; margin-top:6px;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL  (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model…")
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

MODEL = load_model()
PIPELINE    = MODEL["pipeline"]
THRESHOLD   = MODEL["threshold"]
FEATURE_NAMES = MODEL["feature_names"]
NUM_COLS    = MODEL["num_cols"]
CAT_COLS    = MODEL["cat_cols"]
CAT_VALUES  = MODEL["cat_values"]
METRICS     = MODEL["metrics"]
X_TEST      = MODEL["X_test"]
Y_TEST      = MODEL["y_test"]
Y_PROB_TEST = MODEL["y_prob_test"]
DF          = MODEL["df"]

# Derived once
FPR, TPR, _   = roc_curve(Y_TEST, Y_PROB_TEST)
PREC, REC, _  = precision_recall_curve(Y_TEST, Y_PROB_TEST)
CM            = confusion_matrix(Y_TEST, (Y_PROB_TEST >= THRESHOLD).astype(int))

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "primary":   "#3B82F6",
    "success":   "#10B981",
    "danger":    "#EF4444",
    "warning":   "#F59E0B",
    "purple":    "#8B5CF6",
    "bg_card":   "#1E293B",
    "bg_dark":   "#0F172A",
    "text":      "#F1F5F9",
    "muted":     "#64748B",
}

def kpi_card(value, label, color, delta_text=""):
    return f"""
    <div class="kpi-card">
        <div class="kpi-value" style="color:{color};">{value}</div>
        <div class="kpi-label">{label}</div>
        {"<div class='kpi-delta' style='color:"+color+";opacity:.75;'>"+delta_text+"</div>" if delta_text else ""}
    </div>"""

def section(title, icon=""):
    st.markdown(f'<div class="section-header">{icon} {title}</div>', unsafe_allow_html=True)

def alert(text, kind="info"):
    st.markdown(f'<div class="alert-box alert-{kind}">{text}</div>', unsafe_allow_html=True)

def risk_badge(label):
    return f'<span class="risk-{label.lower()}">{label.upper()}</span>'

def get_risk(prob):
    if prob >= 0.7: return "TINGGI",  COLORS["danger"]
    if prob >= 0.4: return "SEDANG", COLORS["warning"]
    return "RENDAH", COLORS["success"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – HOME & OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
def page_home():
    # ── Hero ──
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1E3A8A 0%,#1E293B 60%,#0F172A 100%);
                border-radius:16px;padding:40px 44px;margin-bottom:28px;
                border:1px solid #2563EB;box-shadow:0 8px 40px rgba(37,99,235,0.2);">
        <div style="font-size:0.75rem;color:#93C5FD;letter-spacing:2px;text-transform:uppercase;
                    margin-bottom:8px;">🔮 SISTEM PRODUKSI · AKTIF</div>
        <h1 style="font-size:2.4rem;font-weight:900;color:#F1F5F9;margin:0 0 10px 0;">
            Platform Prediksi Churn Pelanggan E-Commerce
        </h1>
        <p style="color:#94A3B8;font-size:1.05rem;max-width:640px;margin:0 0 20px 0;">
            Platform machine learning yang mengidentifikasi pelanggan berisiko secara real-time,
            memungkinkan tim CRM melakukan intervensi retensi sebelum pelanggan benar-benar pergi.
        </p>
        <div style="display:flex;gap:10px;flex-wrap:wrap;">
            <span style="background:#1E3A8A;color:#93C5FD;border:1px solid #2563EB;
                         padding:4px 14px;border-radius:6px;font-size:0.78rem;font-weight:600;">
                ✅ Model: LightGBM</span>
            <span style="background:#052E16;color:#6EE7B7;border:1px solid #10B981;
                         padding:4px 14px;border-radius:6px;font-size:0.78rem;font-weight:600;">
                🎯 Recall 96.8%</span>
            <span style="background:#1C1000;color:#FDE68A;border:1px solid #F59E0B;
                         padding:4px 14px;border-radius:6px;font-size:0.78rem;font-weight:600;">
                📊 PR-AUC 0.9983</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ──
    section("Metrik Performa Sistem", "📈")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = [
        (c1, f"{METRICS['accuracy']*100:.1f}%", "Akurasi",    COLORS["primary"],  "Set uji (n=1.126)"),
        (c2, f"{METRICS['recall']*100:.1f}%",   "Recall",     COLORS["success"],  "Tingkat deteksi churn"),
        (c3, f"{METRICS['precision']*100:.1f}%","Presisi",    COLORS["warning"],  "Akurasi saat diprediksi churn"),
        (c4, f"{METRICS['pr_auc']:.4f}",         "PR-AUC",     COLORS["danger"],   "Kurva Precision-Recall"),
        (c5, f"{METRICS['f1']*100:.1f}%",        "F1-Score",   COLORS["purple"],   "Rata-rata harmonik P & R"),
    ]
    for col, val, lbl, clr, dlt in kpis:
        col.markdown(kpi_card(val, lbl, clr, dlt), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: business context + usage guide ──
    col_a, col_b = st.columns([1.1, 0.9], gap="large")

    with col_a:
        section("Konteks Bisnis", "🏢")
        st.markdown("""
        <div style="background:#1E293B;border-radius:12px;padding:20px 24px;border:1px solid #334155;">
        <p style="color:#CBD5E1;line-height:1.7;">
            Mendapatkan pelanggan baru di e-commerce biayanya <strong style="color:#F59E0B;">5–7×</strong>
            lebih mahal dibanding mempertahankan pelanggan lama.  Platform ini menilai
            <strong style="color:#3B82F6;">seluruh pelanggan aktif</strong> dan menampilkan mereka yang
            paling berisiko pergi dalam 30 hari ke depan, memberi cukup waktu bagi tim CRM
            dan Marketing untuk mengintervensi dengan penawaran personal.
        </p>
        <hr style="border-color:#334155;margin:14px 0;">
        <table style="width:100%;border-collapse:collapse;font-size:0.85rem;color:#94A3B8;">
            <tr><td style="padding:6px 0;"><strong style="color:#3B82F6;">Dataset</strong></td>
                <td>E-Commerce India — 5.630 pelanggan</td></tr>
            <tr><td style="padding:6px 0;"><strong style="color:#3B82F6;">Tingkat churn dasar</strong></td>
                <td>16,8 % (948 dari 5.630)</td></tr>
            <tr><td style="padding:6px 0;"><strong style="color:#3B82F6;">Decision threshold</strong></td>
                <td>{:.3f} (dioptimalkan via OOF CV)</td></tr>
            <tr><td style="padding:6px 0;"><strong style="color:#3B82F6;">Jadwal retraining</strong></td>
                <td>Bulanan (rolling window 12 bulan)</td></tr>
            <tr><td style="padding:6px 0;"><strong style="color:#3B82F6;">Latensi inferensi</strong></td>
                <td>&lt; 50 ms per pelanggan</td></tr>
        </table>
        </div>
        """.format(THRESHOLD), unsafe_allow_html=True)

    with col_b:
        section("Siapa Pengguna Platform Ini", "👥")
        users = [
            ("🎯", "Tim Marketing",    "#2563EB", "Kampanye retensi bulanan – segmentasi pelanggan risiko rendah / sedang / tinggi ke daftar komunikasi yang tepat sasaran."),
            ("📞", "CRM / Support",    "#059669", "Pemantauan mingguan – hubungi pelanggan secara proaktif sebelum mereka memutuskan untuk pergi."),
            ("🔧", "Layanan Pelanggan","#D97706", "Triase pasca-komplain – langsung tandai pelanggan berisiko tinggi setelah komplain masuk."),
            ("📊", "Manajemen",        "#7C3AED", "Perencanaan strategis – review faktor pendorong churn dan ROI program retensi."),
        ]
        for icon, role, color, desc in users:
            st.markdown(f"""
            <div style="background:#0F172A;border:1px solid #1E293B;border-left:3px solid {color};
                        border-radius:8px;padding:12px 16px;margin-bottom:10px;">
                <div style="font-size:0.9rem;font-weight:700;color:{color};">{icon} {role}</div>
                <div style="font-size:0.80rem;color:#64748B;margin-top:4px;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── How it works ──
    st.markdown("<br>", unsafe_allow_html=True)
    section("Cara Kerja Sistem", "⚙️")
    steps = [
        ("1", "Ingest Data", "Data perilaku pelanggan diambil dari sistem CRM & manajemen pesanan.", "#3B82F6"),
        ("2", "Rekayasa Fitur", "13 fitur numerik + 5 fitur kategorik diproses melalui pipeline MICE imputer + scaler.", "#8B5CF6"),
        ("3", "Penilaian Churn", f"LightGBM menghasilkan skor probabilitas; pelanggan di atas threshold {THRESHOLD:.3f} ditandai berisiko.", "#F59E0B"),
        ("4", "Segmentasi Risiko", "Skor dibagi ke tiga tier: RENDAH / SEDANG / TINGGI untuk prioritas intervensi.", "#10B981"),
        ("5", "Aksi & Umpan Balik", "Tindakan CRM dicatat; hasilnya menjadi input retraining bulanan berikutnya.", "#EF4444"),
    ]
    cols = st.columns(5)
    for col, (num, title, desc, color) in zip(cols, steps):
        col.markdown(f"""
        <div style="background:#1E293B;border-radius:12px;padding:18px 16px;height:100%;
                    border-top:3px solid {color};border:1px solid #334155;
                    box-shadow:0 2px 12px rgba(0,0,0,.3);">
            <div style="font-size:1.6rem;font-weight:900;color:{color};margin-bottom:6px;">{num}</div>
            <div style="font-size:0.9rem;font-weight:700;color:#F1F5F9;margin-bottom:8px;">{title}</div>
            <div style="font-size:0.78rem;color:#64748B;line-height:1.55;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Confusion matrix quick view ──
    st.markdown("<br>", unsafe_allow_html=True)
    section("Evaluasi Terbaru – Ringkasan Confusion Matrix", "🗂️")
    TP, FP, FN, TN = METRICS["TP"], METRICS["FP"], METRICS["FN"], METRICS["TN"]
    total = TP + FP + FN + TN
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, f"{TP}",  "True Positives",  COLORS["success"], f"Churn berhasil ditangkap ({TP/total*100:.1f}%)"),
        (c2, f"{TN}",  "True Negatives",  COLORS["primary"], f"Pelanggan loyal terkonfirmasi ({TN/total*100:.1f}%)"),
        (c3, f"{FP}",  "False Positives", COLORS["warning"], f"Biaya retensi tidak perlu"),
        (c4, f"{FN}",  "False Negatives", COLORS["danger"],  f"Churn yang terlewat"),
    ]
    for col, val, lbl, clr, dlt in cards:
        col.markdown(kpi_card(val, lbl, clr, dlt), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – EDA & INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
def page_eda():
    st.markdown('<h2 style="color:#F1F5F9;margin-bottom:4px;">📊 EDA & Wawasan Pelanggan</h2>', unsafe_allow_html=True)
    alert("Analisis eksplorasi pada dataset lengkap (5.630 pelanggan · 18 fitur).", "info")

    churn_val = DF["Churn"].value_counts()
    churn_pct = DF["Churn"].mean() * 100

    # Row 1 – churn distribution
    col1, col2 = st.columns(2, gap="large")
    with col1:
        section("Distribusi Kelas Churn", "🎯")
        fig = go.Figure(go.Pie(
            labels=["Tidak Churn", "Churn"],
            values=[churn_val[0], churn_val[1]],
            hole=0.62,
            marker_colors=[COLORS["primary"], COLORS["danger"]],
            textinfo="percent+label",
            textfont_color="white",
        ))
        fig.add_annotation(text=f"<b>{churn_pct:.1f}%</b><br>Churn",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=20, color="white"))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=True, legend=dict(font=dict(color="white")),
            height=320, margin=dict(t=10, b=10, l=0, r=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section("Churn Berdasarkan Tenure (bulan)", "📅")
        tenure_churn = DF.groupby("Churn")["Tenure"].apply(list)
        fig = go.Figure()
        for label, color in [(0, COLORS["primary"]), (1, COLORS["danger"])]:
            fig.add_trace(go.Histogram(
                x=tenure_churn[label], name="Tidak Churn" if label==0 else "Churn",
                marker_color=color, opacity=0.75, nbinsx=25,
            ))
        fig.update_layout(
            barmode="overlay", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Tenure (bulan)", color="white", gridcolor="#1E293B"),
            yaxis=dict(title="Jumlah", color="white", gridcolor="#1E293B"),
            legend=dict(font=dict(color="white")),
            height=320, margin=dict(t=10, b=40, l=40, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Row 2 – categorical features
    section("Tingkat Churn Berdasarkan Fitur Kategorik", "📂")
    cat_cols_vis = ["PreferredLoginDevice", "PreferredPaymentMode",
                    "PreferedOrderCat", "MaritalStatus", "Gender"]
    tabs = st.tabs([f"  {c}  " for c in cat_cols_vis])
    for tab, col_name in zip(tabs, cat_cols_vis):
        with tab:
            grp = DF.groupby(col_name)["Churn"].agg(["mean", "sum", "count"]).reset_index()
            grp.columns = [col_name, "Tingkat Churn", "Churn", "Total"]
            grp["Tingkat Churn %"] = (grp["Tingkat Churn"] * 100).round(1)
            grp = grp.sort_values("Tingkat Churn %", ascending=False)
            figc = px.bar(
                grp, x=col_name, y="Tingkat Churn %",
                color="Tingkat Churn %",
                color_continuous_scale=[[0,"#10B981"],[0.5,"#F59E0B"],[1,"#EF4444"]],
                text="Tingkat Churn %",
                hover_data=["Churn", "Total"],
            )
            figc.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            figc.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="white", gridcolor="#1E293B"),
                yaxis=dict(color="white", gridcolor="#1E293B"),
                coloraxis_showscale=False,
                height=360, margin=dict(t=20, b=40, l=40, r=10),
                font=dict(color="white"),
            )
            st.plotly_chart(figc, use_container_width=True)

    # Row 3 – numerical vs churn box plots
    section("Distribusi Fitur Numerik vs Churn", "📉")
    num_features_sel = st.multiselect(
        "Pilih fitur untuk dibandingkan",
        options=NUM_COLS,
        default=["Tenure", "CashbackAmount", "Complain", "SatisfactionScore", "DaySinceLastOrder"],
        key="eda_num",
    )
    if num_features_sel:
        fig_box = make_subplots(rows=1, cols=len(num_features_sel),
                                subplot_titles=num_features_sel)
        for i, feat in enumerate(num_features_sel, 1):
            for churn_val_idx, name, color in [(0, "Tidak Churn", COLORS["primary"]),
                                               (1, "Churn",      COLORS["danger"])]:
                data = DF[DF["Churn"] == churn_val_idx][feat].dropna()
                fig_box.add_trace(
                    go.Box(y=data, name=name, marker_color=color,
                           showlegend=(i==1), legendgroup=name),
                    row=1, col=i,
                )
        fig_box.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            height=400, font=dict(color="white"),
            legend=dict(font=dict(color="white")),
            margin=dict(t=40, b=20, l=40, r=10),
        )
        fig_box.update_xaxes(color="white")
        fig_box.update_yaxes(color="white", gridcolor="#1E293B")
        st.plotly_chart(fig_box, use_container_width=True)

    # Row 4 – key findings
    section("Temuan Utama", "🔑")
    findings = [
        ("📅", "Tenure adalah Faktor #1",
         "Pelanggan dengan tenure < 4 bulan <b>3,4×</b> lebih berisiko churn. "
         "Masa orientasi 90 hari pertama adalah jendela retensi paling kritis.",
         COLORS["danger"]),
        ("⚠️", "Komplain Melipatgandakan Risiko Churn",
         "Pelanggan yang pernah komplain churn di angka <b>31,7%</b> vs hanya <b>10,9%</b> "
         "bagi yang belum pernah komplain — kenaikan 2,9×.",
         COLORS["warning"]),
        ("💰", "Cashback Meningkatkan Loyalitas",
         "Jumlah cashback yang lebih tinggi berkorelasi kuat dengan tingkat churn lebih rendah. "
         "Program cashback bertier bisa jadi tuas retensi yang efektif.",
         COLORS["success"]),
        ("🏙️", "Kota Tier 3 Paling Rentan",
         "Pelanggan kota tier 3 (kota kecil) lebih sensitif terhadap churn, "
         "kemungkinan akibat loyalitas merek yang rendah dan lebih banyak pilihan alternatif.",
         COLORS["primary"]),
    ]
    cols = st.columns(4)
    for col, (icon, title, body, color) in zip(cols, findings):
        col.markdown(f"""
        <div style="background:#1E293B;border-radius:12px;padding:16px;height:100%;
                    border-left:3px solid {color};border:1px solid #334155;">
            <div style="font-size:1.4rem;">{icon}</div>
            <div style="font-size:0.88rem;font-weight:700;color:{color};margin:6px 0 8px;">{title}</div>
            <div style="font-size:0.78rem;color:#94A3B8;line-height:1.6;">{body}</div>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
def page_model():
    st.markdown('<h2 style="color:#F1F5F9;margin-bottom:4px;">🤖 Performa Model</h2>',
                unsafe_allow_html=True)
    alert(
        f"Evaluasi pada test set (n = {len(Y_TEST)} pelanggan · "
        f"decision threshold dioptimalkan via Out-of-Fold CV ke {THRESHOLD:.3f}).",
        "info",
    )

    # ── Metric KPIs ──
    section("Metrik Test Set", "📐")
    cols = st.columns(6)
    metrics_disp = [
        ("PR-AUC",    f"{METRICS['pr_auc']:.4f}",         COLORS["danger"]),
        ("Recall",    f"{METRICS['recall']*100:.2f}%",    COLORS["success"]),
        ("Presisi",   f"{METRICS['precision']*100:.2f}%", COLORS["warning"]),
        ("F1-Score",  f"{METRICS['f1']*100:.2f}%",        COLORS["purple"]),
        ("Akurasi",   f"{METRICS['accuracy']*100:.2f}%",  COLORS["primary"]),
        ("ROC-AUC",   f"{METRICS['roc_auc']:.4f}",        "#06B6D4"),
    ]
    for col, (lbl, val, clr) in zip(cols, metrics_disp):
        col.markdown(kpi_card(val, lbl, clr), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CM + ROC ──
    col1, col2 = st.columns(2, gap="large")

    with col1:
        section("Confusion Matrix", "🔲")
        TN, FP, FN, TP = CM.ravel()
        z     = [[TN, FP], [FN, TP]]
        x_lbl = ["Prediksi: Tidak Churn", "Prediksi: Churn"]
        y_lbl = ["Aktual: Tidak Churn",   "Aktual: Churn"]
        text  = [[f"TN={TN}", f"FP={FP}"], [f"FN={FN}", f"TP={TP}"]]
        fig_cm = go.Figure(go.Heatmap(
            z=z, x=x_lbl, y=y_lbl,
            text=text, texttemplate="%{text}<br>(%{z})",
            colorscale=[[0,"#1E293B"],[0.5,"#2563EB"],[1,"#7C3AED"]],
            showscale=False, textfont=dict(color="white", size=14),
        ))
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="white"), yaxis=dict(color="white"),
            height=320, margin=dict(t=10, b=40, l=100, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col2:
        section("Kurva Precision-Recall (PR)", "🎯")
        no_skill = Y_TEST.mean()
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(
            x=[0,1], y=[no_skill, no_skill], mode="lines", name="No-Skill",
            line=dict(dash="dash", color=COLORS["muted"], width=1.5),
        ))
        fig_pr.add_trace(go.Scatter(
            x=REC, y=PREC, mode="lines",
            name=f"LightGBM  PR-AUC={METRICS['pr_auc']:.4f}",
            line=dict(color=COLORS["danger"], width=2.5),
            fill="tozeroy", fillcolor="rgba(239,68,68,0.10)",
        ))
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Recall",    color="white", gridcolor="#1E293B"),
            yaxis=dict(title="Precision", color="white", gridcolor="#1E293B"),
            legend=dict(font=dict(color="white")),
            height=320, margin=dict(t=10, b=40, l=60, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── PR Curve + Score Distribution ──
    col3, col4 = st.columns(2, gap="large")

    with col3:
        section("Kurva ROC", "📈")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0,1], y=[0,1], mode="lines", name="Acak",
            line=dict(dash="dash", color=COLORS["muted"], width=1.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=FPR, y=TPR, mode="lines",
            name=f"LightGBM  ROC-AUC={METRICS['roc_auc']:.4f}",
            line=dict(color=COLORS["primary"], width=2.5),
            fill="tozeroy", fillcolor="rgba(59,130,246,0.10)",
        ))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="False Positive Rate", color="white", gridcolor="#1E293B"),
            yaxis=dict(title="True Positive Rate",  color="white", gridcolor="#1E293B"),
            legend=dict(font=dict(color="white")),
            height=320, margin=dict(t=10, b=40, l=60, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col4:
        section(f"Distribusi Skor (threshold = {THRESHOLD:.3f})", "📊")
        churn_probs      = Y_PROB_TEST[Y_TEST == 1]
        not_churn_probs  = Y_PROB_TEST[Y_TEST == 0]
        fig_dist = go.Figure()
        for probs, name, color in [(not_churn_probs, "Tidak Churn", COLORS["primary"]),
                                   (churn_probs,     "Churn",       COLORS["danger"])]:
            fig_dist.add_trace(go.Histogram(
                x=probs, name=name, nbinsx=30,
                marker_color=color, opacity=0.72,
            ))
        fig_dist.add_vline(
            x=THRESHOLD, line_dash="dash",
            line_color=COLORS["warning"], line_width=2,
            annotation_text=f"Threshold {THRESHOLD:.3f}",
            annotation_font_color=COLORS["warning"],
        )
        fig_dist.update_layout(
            barmode="overlay",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Probabilitas Churn", color="white", gridcolor="#1E293B"),
            yaxis=dict(title="Jumlah",             color="white", gridcolor="#1E293B"),
            legend=dict(font=dict(color="white")),
            height=320, margin=dict(t=10, b=40, l=40, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # ── Feature Importance ──
    section("Kepentingan Fitur (LightGBM Gain)", "🏆")
    try:
        booster  = PIPELINE.named_steps["model"].booster_
        imp      = booster.feature_importance(importance_type="gain")
        feat_names_encoded = PIPELINE.named_steps["preprocessing"] \
                                     .get_feature_names_out()
        raw_names = [n.split("__")[-1] for n in feat_names_encoded]
        imp_df = pd.DataFrame({"Feature": raw_names, "Importance": imp})
        imp_df = imp_df.groupby("Feature")["Importance"].sum().reset_index()
        imp_df = imp_df.sort_values("Importance", ascending=True).tail(15)
        max_val = imp_df["Importance"].max()
        imp_df["Color"] = imp_df["Importance"].apply(
            lambda v: COLORS["danger"] if v > 0.7*max_val
                      else (COLORS["warning"] if v > 0.4*max_val else COLORS["primary"])
        )
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"], y=imp_df["Feature"],
            orientation="h",
            marker_color=imp_df["Color"].tolist(),
            text=imp_df["Importance"].apply(lambda v: f"{v:,.0f}"),
            textposition="outside", textfont=dict(color="white",size=10),
        ))
        fig_imp.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="white", gridcolor="#1E293B", title="Gain"),
            yaxis=dict(color="white"),
            height=440, margin=dict(t=10, b=40, l=160, r=60),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception as e:
        alert(f"Grafik kepentingan fitur tidak tersedia: {e}", "warning")

    section("Laporan Klasifikasi", "📋")
    report_data = {
        "Kelas":     ["Tidak Churn", "Churn", "", "Akurasi", "Rata-rata Makro", "Rata-rata Tertimbang"],
        "Presisi":   ["0.993", f"{METRICS['precision']:.3f}", "", f"{METRICS['accuracy']:.3f}", "0.989", "0.992"],
        "Recall":    ["0.997", f"{METRICS['recall']:.3f}",   "", f"{METRICS['accuracy']:.3f}", "0.982", "0.992"],
        "F1-Score":  ["0.995", f"{METRICS['f1']:.3f}",      "", f"{METRICS['accuracy']:.3f}", "0.986", "0.992"],
        "Support":   ["936",   "190", "", "1126", "1126", "1126"],
    }
    rdf = pd.DataFrame(report_data)
    st.dataframe(
        rdf.style.set_properties(**{"background-color": "#1E293B", "color": "#F1F5F9",
                                    "border-color": "#334155"}),
        use_container_width=True, hide_index=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 – PREDICT CHURN
# ─────────────────────────────────────────────────────────────────────────────
def page_predict():
    st.markdown('<h2 style="color:#F1F5F9;margin-bottom:4px;">🎯 Prediksi Churn Pelanggan</h2>',
                unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["  Pelanggan Tunggal  ", "  Upload Batch (CSV)  "])

    # ── TAB 1 – single ──────────────────────────────────────────────────────
    with tab1:
        alert(
            "Isi profil pelanggan di bawah lalu klik <b>Prediksi</b> untuk mendapatkan skor risiko "
            "churn secara real-time beserta rekomendasi tindakan retensi.",
            "info",
        )
        with st.form("single_pred_form"):
            st.markdown('<div class="section-header">📋 Profil Pelanggan</div>',
                        unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3, gap="large")

            with col_a:
                st.markdown("**Demografi & Akun**")
                tenure       = st.number_input("Tenure (bulan)", 0, 72, 12, step=1)
                city_tier    = st.selectbox("Tier Kota", [1, 2, 3], index=1)
                gender       = st.selectbox("Jenis Kelamin", CAT_VALUES["Gender"])
                marital_st   = st.selectbox("Status Pernikahan", CAT_VALUES["MaritalStatus"])
                num_address  = st.number_input("Jumlah Alamat", 1, 20, 2, step=1)

            with col_b:
                st.markdown("**Perilaku Aplikasi & Perangkat**")
                login_device = st.selectbox("Perangkat Login Utama",
                                            CAT_VALUES["PreferredLoginDevice"])
                hour_app     = st.number_input("Jam di Aplikasi (per hari)", 0.0, 10.0, 3.0, step=0.5)
                num_devices  = st.number_input("Perangkat Terdaftar", 1, 10, 3, step=1)
                order_cat    = st.selectbox("Kategori Pesanan Favorit",
                                            CAT_VALUES["PreferedOrderCat"])
                pay_mode     = st.selectbox("Metode Pembayaran Utama",
                                            CAT_VALUES["PreferredPaymentMode"])

            with col_c:
                st.markdown("**Metrik Pesanan & Kepuasan**")
                satisfaction  = st.slider("Skor Kepuasan", 1, 5, 3)
                warehouse_km  = st.number_input("Jarak Gudang ke Rumah (km)", 5, 150, 30, step=5)
                complain      = st.selectbox("Pernah Komplain?", ["Tidak (0)", "Ya (1)"])
                complain_val  = int(complain.split("(")[1].rstrip(")"))
                order_hike    = st.number_input("Kenaikan Nilai Pesanan dari Tahun Lalu (%)",
                                                0.0, 100.0, 15.0, step=1.0)
                coupon_used   = st.number_input("Kupon Digunakan (bulan terakhir)", 0, 20, 1, step=1)
                order_count   = st.number_input("Pesanan (bulan terakhir)", 1, 30, 3, step=1)
                days_last     = st.number_input("Hari Sejak Pesanan Terakhir", 0, 60, 7, step=1)
                cashback      = st.number_input("Jumlah Cashback (₹)", 0.0, 400.0, 150.0, step=10.0)

            submitted = st.form_submit_button("🔮  Jalankan Prediksi Churn", use_container_width=True)

        if submitted:
            input_data = pd.DataFrame([{
                "Tenure":                   tenure,
                "PreferredLoginDevice":     login_device,
                "CityTier":                 city_tier,
                "WarehouseToHome":          warehouse_km,
                "PreferredPaymentMode":     pay_mode,
                "Gender":                   gender,
                "HourSpendOnApp":           hour_app,
                "NumberOfDeviceRegistered": num_devices,
                "PreferedOrderCat":         order_cat,
                "SatisfactionScore":        satisfaction,
                "MaritalStatus":            marital_st,
                "NumberOfAddress":          num_address,
                "Complain":                 complain_val,
                "OrderAmountHikeFromlastYear": order_hike,
                "CouponUsed":               coupon_used,
                "OrderCount":               order_count,
                "DaySinceLastOrder":        days_last,
                "CashbackAmount":           cashback,
            }])

            prob  = float(PIPELINE.predict_proba(input_data)[0, 1])
            pred  = int(prob >= THRESHOLD)
            risk, risk_color = get_risk(prob)

            # Result panel
            st.markdown("<br>", unsafe_allow_html=True)
            section("Hasil Prediksi", "🔮")
            cr1, cr2, cr3 = st.columns([1, 1.2, 1], gap="large")

            # Gauge
            with cr1:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(prob * 100, 1),
                    number=dict(suffix="%", font=dict(color="white", size=44)),
                    gauge=dict(
                        axis=dict(range=[0, 100], tickcolor="white",
                                  tickfont=dict(color="white")),
                        bar=dict(color=risk_color, thickness=0.28),
                        bgcolor="#1E293B",
                        borderwidth=0,
                        steps=[
                            dict(range=[0, 40],  color="#052E16"),
                            dict(range=[40, 70], color="#78350F"),
                            dict(range=[70, 100],color="#450A0A"),
                        ],
                        threshold=dict(
                            line=dict(color=COLORS["warning"], width=3),
                            thickness=0.85,
                            value=round(THRESHOLD * 100, 1),
                        ),
                    ),
                    title=dict(text="Probabilitas Churn", font=dict(color="#94A3B8", size=14)),
                ))
                fig_gauge.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", height=260,
                    margin=dict(t=20, b=0, l=30, r=30),
                    font=dict(color="white"),
                )
                st.plotly_chart(fig_gauge, use_container_width=True)

            # Verdict
            with cr2:
                verdict_class = "pred-churn" if pred else "pred-nochurn"
                verdict_icon  = "🔴" if pred else "🟢"
                verdict_text  = "RISIKO CHURN TERDETEKSI" if pred else "PELANGGAN STABIL"
                verdict_sub   = (f"Probabilitas {prob*100:.1f}% melebihi threshold "
                                 f"{THRESHOLD*100:.1f}%  →  Intervensi diperlukan." if pred
                                 else f"Probabilitas {prob*100:.1f}% di bawah threshold "
                                      f"{THRESHOLD*100:.1f}%  →  Tidak perlu tindakan.")
                st.markdown(f"""
                <div class="pred-panel {verdict_class}">
                    <div style="font-size:2.4rem;margin-bottom:8px;">{verdict_icon}</div>
                    <div class="pred-score" style="color:{'#EF4444' if pred else '#10B981'};">
                        {pred and 'BERISIKO' or 'AMAN'}</div>
                    <div style="font-size:1rem;font-weight:700;color:white;margin:8px 0;">
                        {verdict_text}</div>
                    <div class="pred-subtitle">{verdict_sub}</div>
                    <div style="margin-top:14px;">{risk_badge(risk)}</div>
                </div>""", unsafe_allow_html=True)

            # Rekomendasi
            with cr3:
                section("Rekomendasi Tindakan", "⚡")
                if pred:
                    actions = {
                        "TINGGI": [
                            ("🎁", "Tawarkan cashback VIP segera"),
                            ("📞", "Hubungi CS prioritas dalam 24 jam"),
                            ("💳", "Aktifkan upgrade tier loyalitas"),
                            ("📧", "Kirim email re-engagement personal"),
                        ],
                        "SEDANG": [
                            ("🎫", "Kirim kode kupon bertarget"),
                            ("📱", "Push notification dengan penawaran"),
                            ("⭐", "Survei tindak lanjut kepuasan"),
                        ],
                    }.get(risk, [("📊", "Pantau setiap minggu")])
                    for icon, action in actions:
                        st.markdown(f"""
                        <div style="background:#0F172A;border:1px solid #1E293B;
                                    border-left:3px solid {risk_color};border-radius:6px;
                                    padding:8px 12px;margin-bottom:6px;font-size:0.82rem;
                                    color:#CBD5E1;">
                            {icon} {action}</div>""", unsafe_allow_html=True)
                else:
                    actions_ok = [
                        ("✅", "Lanjutkan keterlibatan standar"),
                        ("📊", "Pantau pada siklus berikutnya"),
                        ("🎯", "Peluang upsell / cross-sell"),
                    ]
                    for icon, action in actions_ok:
                        st.markdown(f"""
                        <div style="background:#0F172A;border:1px solid #1E293B;
                                    border-left:3px solid #10B981;border-radius:6px;
                                    padding:8px 12px;margin-bottom:6px;font-size:0.82rem;
                                    color:#CBD5E1;">
                            {icon} {action}</div>""", unsafe_allow_html=True)

            # Profile summary table
            st.markdown("<br>", unsafe_allow_html=True)
            section("Ringkasan Input", "📄")
            disp = input_data.T.reset_index()
            disp.columns = ["Fitur", "Nilai"]
            st.dataframe(
                disp.style.set_properties(**{"background-color": "#1E293B",
                                             "color": "#F1F5F9", "border-color": "#334155"}),
                use_container_width=True, hide_index=True, height=320,
            )

    # ── TAB 2 – batch ───────────────────────────────────────────────────────
    with tab2:
        alert(
            "Upload file CSV dengan header kolom yang sama seperti data pelatihan. "
            "Hasil akan memiliki kolom <b>ProbabilitasChurn</b>, <b>PrediksiChurn</b>, "
            "dan <b>TierRisiko</b>.",
            "info",
        )

        sample = pd.DataFrame([{f: "" for f in FEATURE_NAMES}])
        csv_bytes = sample.to_csv(index=False).encode()
        st.download_button(
            "⬇️  Unduh Template CSV",
            csv_bytes, "template_churn.csv", "text/csv",
            use_container_width=False,
        )

        uploaded = st.file_uploader("Upload CSV Pelanggan", type="csv", key="batch_upload")
        if uploaded:
            try:
                df_upload = pd.read_csv(uploaded)
                missing_cols = [c for c in FEATURE_NAMES if c not in df_upload.columns]
                if missing_cols:
                    alert(f"Kolom tidak ditemukan: {missing_cols}", "danger")
                else:
                    df_input = df_upload[FEATURE_NAMES].copy()
                    probs    = PIPELINE.predict_proba(df_input)[:, 1]
                    preds    = (probs >= THRESHOLD).astype(int)
                    tiers    = ["TINGGI" if p >= 0.7 else "SEDANG" if p >= 0.4 else "RENDAH"
                                for p in probs]
                    df_result = df_upload.copy()
                    df_result["ProbabilitasChurn"] = probs.round(4)
                    df_result["PrediksiChurn"]     = preds
                    df_result["TierRisiko"]        = tiers

                    n_high   = (np.array(tiers) == "TINGGI").sum()
                    n_med    = (np.array(tiers) == "SEDANG").sum()
                    n_low    = (np.array(tiers) == "RENDAH").sum()
                    n_churn  = int(preds.sum())
                    total    = len(df_result)

                    section("Ringkasan Batch", "📊")
                    bc1, bc2, bc3, bc4 = st.columns(4)
                    bc1.markdown(kpi_card(total,   "Pelanggan",    COLORS["primary"]), unsafe_allow_html=True)
                    bc2.markdown(kpi_card(n_churn, "Berisiko",     COLORS["danger"]),  unsafe_allow_html=True)
                    bc3.markdown(kpi_card(n_med,   "Risiko Sedang",COLORS["warning"]), unsafe_allow_html=True)
                    bc4.markdown(kpi_card(n_low,   "Risiko Rendah",COLORS["success"]), unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)
                    section("Pratinjau Hasil (Top 50)", "📋")

                    def color_risk(val):
                        if val == "TINGGI": return "background-color:#7F1D1D;color:#FECACA"
                        if val == "SEDANG": return "background-color:#78350F;color:#FDE68A"
                        return "background-color:#052E16;color:#BBF7D0"

                    styled = (df_result.head(50)
                              .style
                              .applymap(color_risk, subset=["TierRisiko"])
                              .set_properties(**{"background-color": "#1E293B",
                                                 "color": "#F1F5F9", "border-color": "#334155"}))
                    st.dataframe(styled, use_container_width=True, height=400)

                    csv_out = df_result.to_csv(index=False).encode()
                    st.download_button(
                        "⬇️  Unduh Hasil Lengkap CSV",
                        csv_out, "prediksi_churn.csv", "text/csv",
                        use_container_width=False,
                    )
            except Exception as e:
                alert(f"Kesalahan memproses file: {e}", "danger")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 – BUSINESS IMPACT
# ─────────────────────────────────────────────────────────────────────────────
def page_impact():
    st.markdown('<h2 style="color:#F1F5F9;margin-bottom:4px;">💰 Simulasi Dampak Bisnis</h2>',
                unsafe_allow_html=True)
    alert(
        "Sesuaikan parameter biaya untuk mensimulasikan seberapa besar model mengurangi "
        "pengeluaran retensi dibanding skenario tanpa model.",
        "info",
    )

    section("Parameter Simulasi", "⚙️")
    sc1, sc2, sc3 = st.columns(3, gap="large")
    with sc1:
        cac = st.number_input("Biaya Akuisisi Pelanggan – CAC (₹)",
                              100, 10000, 1500, step=100,
                              help="Biaya mendapatkan pelanggan baru untuk mengganti yang churn")
        crc = st.number_input("Biaya Retensi Pelanggan – CRC (₹)",
                              50, 2000, 300, step=50,
                              help="Biaya satu tindakan retensi (kupon, telepon, dll.)")
    with sc2:
        n_customers   = st.number_input("Total Pelanggan dalam Cohort",
                                        100, 100000, 5630, step=100)
        churn_rate    = st.slider("Perkiraan Tingkat Churn (%)", 5.0, 40.0,
                                  float(round(DF["Churn"].mean()*100, 1)), step=0.5)
    with sc3:
        model_recall  = st.slider("Recall Model (%)",  70.0, 100.0,
                                  float(round(METRICS["recall"]*100, 1)), step=0.5)
        model_fp_rate = st.slider("Tingkat False Positive Model (%)", 0.0, 30.0, 0.6, step=0.5)

    # ── Calculations ──
    n_churn      = round(n_customers * churn_rate / 100)
    n_no_churn   = n_customers - n_churn

    # Scenario A – No Action
    cost_no_action = n_churn * cac

    # Scenario B – Mass Retention (treat everyone)
    cost_mass = n_customers * crc

    # Scenario C – Model Targeted
    tp_model = round(n_churn * model_recall / 100)
    fp_model = round(n_no_churn * model_fp_rate / 100)
    fn_model = n_churn - tp_model
    cost_model = (tp_model + fp_model) * crc + fn_model * cac
    saving_vs_noaction = max(0, cost_no_action - cost_model)
    saving_pct = saving_vs_noaction / cost_no_action * 100 if cost_no_action else 0

    # ── KPI Row ──
    st.markdown("<br>", unsafe_allow_html=True)
    section("Perbandingan Biaya Antar Skenario", "📊")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(kpi_card(f"₹{cost_no_action:,.0f}", "Biaya Tanpa Aksi",
                         COLORS["danger"], f"{n_churn} churn × ₹{cac}"), unsafe_allow_html=True)
    k2.markdown(kpi_card(f"₹{cost_mass:,.0f}", "Biaya Retensi Massal",
                         COLORS["warning"], f"{n_customers} pelanggan × ₹{crc}"), unsafe_allow_html=True)
    k3.markdown(kpi_card(f"₹{cost_model:,.0f}", "Biaya Bertarget Model",
                         COLORS["success"], f"TP={tp_model} FP={fp_model} FN={fn_model}"), unsafe_allow_html=True)
    k4.markdown(kpi_card(f"₹{saving_vs_noaction:,.0f}", "Penghematan vs Tanpa Aksi",
                         COLORS["primary"], f"{saving_pct:.1f}% penurunan biaya"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_chart, col_detail = st.columns([1.2, 0.8], gap="large")

    with col_chart:
        section("Grafik Perbandingan Biaya", "📈")
        scenarios = ["Tanpa Aksi", "Retensi Massal", "Bertarget Model"]
        costs     = [cost_no_action, cost_mass, cost_model]
        colors    = [COLORS["danger"], COLORS["warning"], COLORS["success"]]
        fig_bar   = go.Figure(go.Bar(
            x=scenarios, y=costs,
            marker_color=colors, text=[f"₹{c:,.0f}" for c in costs],
            textposition="outside", textfont=dict(color="white", size=13),
        ))
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="white"),
            yaxis=dict(title="Total Biaya (₹)", color="white", gridcolor="#1E293B"),
            height=360, margin=dict(t=30, b=20, l=60, r=10),
            font=dict(color="white"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_detail:
        section("Rincian Detail", "🗂️")
        detail_df = pd.DataFrame({
            "Skenario":          ["Tanpa Aksi", "Retensi Massal", "Bertarget Model"],
            "Total Biaya (₹)":   [f"₹{cost_no_action:,.0f}",
                                  f"₹{cost_mass:,.0f}",
                                  f"₹{cost_model:,.0f}"],
            "Penghematan vs Tanpa Aksi": [
                "—",
                f"₹{max(0,cost_no_action-cost_mass):,.0f}",
                f"₹{saving_vs_noaction:,.0f}  ({saving_pct:.1f}%)",
            ],
        })
        st.dataframe(
            detail_df.style.set_properties(**{"background-color": "#1E293B",
                                              "color": "#F1F5F9", "border-color": "#334155"}),
            use_container_width=True, hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        # ROI indicator
        roi_val = (saving_vs_noaction / max(1, cost_model)) * 100
        st.markdown(f"""
        <div style="background:#1E293B;border-radius:12px;padding:18px;
                    border-left:4px solid {COLORS['success']};border:1px solid #334155;">
            <div style="font-size:0.75rem;color:#64748B;text-transform:uppercase;letter-spacing:1px;">
                Estimasi ROI Penerapan Model</div>
            <div style="font-size:2.2rem;font-weight:900;color:{COLORS['success']};margin:6px 0;">
                {roi_val:.0f}%</div>
            <div style="font-size:0.78rem;color:#94A3B8;">
                Untuk setiap ₹1 yang dikeluarkan untuk retensi bertarget, model menghasilkan
                <strong style="color:{COLORS['success']};">₹{roi_val/100+1:.2f}</strong> dari biaya akuisisi yang berhasil dihindari.
            </div>
        </div>""", unsafe_allow_html=True)

    # ── Pie: How model catches churn ──
    st.markdown("<br>", unsafe_allow_html=True)
    section("Rincian Hasil Model (Estimasi Cohort)", "🥧")
    tn_est = n_no_churn - fp_model
    labels_pie = ["True Positive (Tertangkap)", "False Positive (Tidak Perlu)",
                  "False Negative (Terlewat)", "True Negative (Benar Aman)"]
    values_pie = [tp_model, fp_model, fn_model, tn_est]
    colors_pie = [COLORS["success"], COLORS["warning"], COLORS["danger"], COLORS["primary"]]
    fig_pie = go.Figure(go.Pie(
        labels=labels_pie, values=values_pie,
        marker_colors=colors_pie, hole=0.55,
        textinfo="percent+label", textfont_color="white",
    ))
    fig_pie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(font=dict(color="white")),
        height=350, margin=dict(t=10, b=10, l=0, r=0),
        font=dict(color="white"),
    )
    st.plotly_chart(fig_pie, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="brand-header">
            <div style="font-size:2rem;margin-bottom:4px;">🔮</div>
            <div class="brand-name">Prediksi Churn</div>
            <div class="brand-tag">Platform Intelijen Pelanggan</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Navigation ──
        st.markdown('<p class="nav-section-title">Navigasi</p>', unsafe_allow_html=True)
        pages = {
            "🏠  Beranda & Ikhtisar": "home",
            "📊  EDA & Wawasan":      "eda",
            "🤖  Performa Model":     "model",
            "🎯  Prediksi Churn":     "predict",
            "💰  Dampak Bisnis":      "impact",
        }
        selection = st.radio(
            "", list(pages.keys()),
            label_visibility="collapsed",
            key="nav_radio",
        )
        st.markdown('<hr style="border:none;border-top:1px solid #1E293B;margin:8px 0 16px 0">', unsafe_allow_html=True)
        st.markdown('<p class="nav-section-title">Info Model</p>', unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#0F172A;border-radius:8px;padding:12px 14px;
                    border:1px solid #1E293B;font-size:0.78rem;color:#64748B;">
            <div style="margin-bottom:4px;">
                <span style="color:#94A3B8;">Algoritma:</span> LightGBM</div>
            <div style="margin-bottom:4px;">
                <span style="color:#94A3B8;">Threshold:</span>
                <span style="color:{COLORS['warning']}">{THRESHOLD:.3f}</span></div>
            <div style="margin-bottom:4px;">
                <span style="color:#94A3B8;">Recall:</span>
                <span style="color:{COLORS['success']}">{METRICS['recall']*100:.1f}%</span></div>
            <div style="margin-bottom:4px;">
                <span style="color:#94A3B8;">PR-AUC:</span>
                <span style="color:{COLORS['danger']}">{METRICS['pr_auc']:.4f}</span></div>
            <div>
                <span style="color:#94A3B8;">Test n:</span> {len(Y_TEST)} pelanggan</div>
        </div>
        """, unsafe_allow_html=True)

        return pages[selection]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    page_key = sidebar()
    if   page_key == "home":    page_home()
    elif page_key == "eda":     page_eda()
    elif page_key == "model":   page_model()
    elif page_key == "predict": page_predict()
    elif page_key == "impact":  page_impact()


if __name__ == "__main__":
    main()
