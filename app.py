"""
Telco Customer Churn Prediction App (Streamlit)
===============================================
Project  : Telco Customer Churn Machine Learning
Dataset  : data/processed/data_cleaned.csv (atau data/raw/data_telco_customer_churn.csv)
           Target: Churn ("Yes" = 1 / Churn, "No" = 0 / Tidak Churn)
Model    : models/churn_pipeline.pkl
           (FixedThresholdClassifier wrapping Pipeline[ColumnTransformer, LogisticRegression(C=1)],
            threshold optimal = 0.27 berdasarkan minimum expected cost pada nested OOF)
Fitur UI : 1) Single Prediction (input widget + preset profil + kalkulator dampak biaya retensi)
           2) Batch Prediction (upload CSV + evaluasi metrik PR-AUC/ROC-AUC + simulasi biaya)
           3) Interpretasi Model & Analisis Bisnis (Permutation Importance, Koefisien LogReg,
              Partial Dependence Plot, serta kurva sensitivitas Threshold & Cost)

Jalankan dari root repository:
    pip install "streamlit>=1.40" pandas numpy plotly scikit-learn joblib
    pip install -e .
    streamlit run streamlit_app.py
"""

import pickle
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# =============================================================================
# RESOLUSI PATH PROJECT & SHIM KOMPATIBILITAS UNPICKLING
# =============================================================================
def _find_repo_root() -> Path:
    """Cari direktori root repository (tempat folder data/ atau models/ berada)."""
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    for candidate in [here, Path.cwd(), *here.parents]:
        if (candidate / "models" / "churn_pipeline.pkl").exists() or (
            candidate / "data" / "processed" / "data_cleaned.csv"
        ).exists():
            return candidate
    return Path.cwd()


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Konstanta fitur sesuai src/data_preprocessing.py & models/model_metadata.json
RANDOM_STATE = 34
DEFAULT_THRESHOLD = 0.27
TARGET_COL = "Churn"
ID_COL = "row_id"

NUM_COLS = ["tenure", "MonthlyCharges"]
BINARY_COLS = ["Dependents", "PaperlessBilling"]
ADDON_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
MULTI_CAT_COLS = ["InternetService", "Contract"]
CAT_COLS = [
    "Dependents",
    "OnlineSecurity",
    "OnlineBackup",
    "InternetService",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
]
FEATURE_COLS = [
    "Dependents",
    "tenure",
    "OnlineSecurity",
    "OnlineBackup",
    "InternetService",
    "DeviceProtection",
    "TechSupport",
    "Contract",
    "PaperlessBilling",
    "MonthlyCharges",
]

CATEGORY_ORDER = {
    "Dependents": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
}
ADDON_ALLOWED = {"No", "Yes", "No internet service"}

# Parameter asumsi model biaya bisnis (src/evaluation.py & reports/model_selection.json)
DEFAULT_HORIZON_MONTHS = 12
DEFAULT_SUCCESS_RATE = 0.30
DEFAULT_OFFER_MONTHS = 1.0


# Fungsi yang direferensikan oleh models/churn_pipeline.pkl saat unpickling
def encode_addons(frame):
    """Add-on services: Yes -> 1, No / No internet service -> 0."""
    frame = pd.DataFrame(frame)
    unexpected = set(pd.unique(frame.to_numpy().ravel())) - ADDON_ALLOWED
    if unexpected:
        raise ValueError(f"Unexpected add-on service values: {unexpected}")
    return frame.eq("Yes").astype(int)


def _log_tenure_name(transformer, input_features):
    return np.array(["tenure_log"])


def _ensure_unpickle_shims():
    """Pastikan modul src.data_preprocessing tersedia meskipun dijalankan secara standalone."""
    try:
        import src.data_preprocessing  # noqa: F401
    except ModuleNotFoundError:
        src_mod = types.ModuleType("src")
        dp_mod = types.ModuleType("src.data_preprocessing")
        dp_mod.encode_addons = encode_addons
        dp_mod._log_tenure_name = _log_tenure_name
        src_mod.data_preprocessing = dp_mod
        sys.modules["src"] = src_mod
        sys.modules["src.data_preprocessing"] = dp_mod


_ensure_unpickle_shims()

# =============================================================================
# KONFIGURASI HALAMAN STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📞",
    layout="wide",
)

CANDIDATE_DATA_PATHS = [
    REPO_ROOT / "data" / "processed" / "data_cleaned.csv",
    REPO_ROOT / "data" / "raw" / "data_telco_customer_churn.csv",
    Path("data/processed/data_cleaned.csv"),
    Path("data/raw/data_telco_customer_churn.csv"),
    Path("data_cleaned.csv"),
    Path("data_telco_customer_churn.csv"),
]

CANDIDATE_MODEL_PATHS = [
    REPO_ROOT / "models" / "churn_pipeline.pkl",
    Path("models/churn_pipeline.pkl"),
    Path("churn_pipeline.pkl"),
]

# Label fitur yang ramah pengguna untuk tampilan UI & grafik
FEATURE_LABELS = {
    "Dependents": "Memiliki Tanggungan (Dependents)",
    "tenure": "Masa Berlangganan / Tenure (bulan)",
    "OnlineSecurity": "Keamanan Online (OnlineSecurity)",
    "OnlineBackup": "Cadangan Online (OnlineBackup)",
    "InternetService": "Jenis Layanan Internet (InternetService)",
    "DeviceProtection": "Proteksi Perangkat (DeviceProtection)",
    "TechSupport": "Dukungan Teknis (TechSupport)",
    "Contract": "Jenis Kontrak (Contract)",
    "PaperlessBilling": "Tagihan Tanpa Kertas (PaperlessBilling)",
    "MonthlyCharges": "Biaya Bulanan / MonthlyCharges ($)",
}

ENCODED_FEATURE_LABELS = {
    "Dependents_Yes": "Dependents = Yes",
    "PaperlessBilling_Yes": "PaperlessBilling = Yes",
    "OnlineSecurity": "OnlineSecurity = Yes",
    "OnlineBackup": "OnlineBackup = Yes",
    "DeviceProtection": "DeviceProtection = Yes",
    "TechSupport": "TechSupport = Yes",
    "InternetService_Fiber optic": "InternetService = Fiber optic (vs DSL)",
    "InternetService_No": "InternetService = No (vs DSL)",
    "Contract_One year": "Contract = One year (vs Month-to-month)",
    "Contract_Two year": "Contract = Two year (vs Month-to-month)",
    "tenure_log": "log(1 + tenure) [standardized]",
    "MonthlyCharges": "MonthlyCharges [standardized]",
}

# Bobot eksak dari models/churn_pipeline.pkl (LogisticRegression C=1, L2, random_state=34)
# Digunakan untuk visualisasi koefisien & sebagai fallback analitik jika versi library berbeda
EXACT_MODEL_WEIGHTS = {
    "encoded_features": [
        "Dependents_Yes",
        "PaperlessBilling_Yes",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "InternetService_Fiber optic",
        "InternetService_No",
        "Contract_One year",
        "Contract_Two year",
        "tenure_log",
        "MonthlyCharges",
    ],
    "coef": [
        -0.253426637012416,
        0.4118694028363782,
        -0.4730473821378721,
        -0.2964019727007794,
        0.034581826283450655,
        -0.3815384518956561,
        0.385524284167509,
        -0.7467702137319394,
        -0.8605721391232478,
        -1.702233444841366,
        -0.7471695181828578,
        0.5548254449550434,
    ],
    "intercept": -0.8908828324225255,
    "tenure_log_mean": 3.0401894131918104,
    "tenure_log_scale": 1.15341459800356,
    "monthly_mean": 64.88303245436104,
    "monthly_scale": 29.92092461557419,
}


def label(col: str) -> str:
    return FEATURE_LABELS.get(col, col)


# =============================================================================
# PEMUATAN MODEL & PREDIKSI
# =============================================================================
@st.cache_resource(show_spinner=False)
def load_model():
    """
    Muat model final dari models/churn_pipeline.pkl.
    Jika file tidak ditemukan atau terjadi perbedaan versi scikit-learn/numpy di environment,
    kembalikan (None, None) agar fungsi prediksi memakai fallback bobot eksak Logistic Regression.
    """
    _ensure_unpickle_shims()
    for path in CANDIDATE_MODEL_PATHS:
        if path.exists():
            try:
                with open(path, "rb") as fh:
                    model = pickle.load(fh)
                test_row = pd.DataFrame([{
                    "Dependents": "No",
                    "tenure": 2,
                    "OnlineSecurity": "No",
                    "OnlineBackup": "No",
                    "InternetService": "Fiber optic",
                    "DeviceProtection": "No",
                    "TechSupport": "No",
                    "Contract": "Month-to-month",
                    "PaperlessBilling": "Yes",
                    "MonthlyCharges": 85.5,
                }])
                _ = model.predict_proba(test_row)
                return model, str(path)
            except Exception:
                try:
                    model = joblib.load(path)
                    return model, str(path)
                except Exception:
                    pass
    return None, None


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pastikan urutan kolom sesuai FEATURE_COLS dan tipe data bersih.
    Transformasi encoding & scaling sudah berada di dalam Pipeline model.
    """
    return coerce_types(df[FEATURE_COLS].copy())


def exact_logreg_predict_proba(df: pd.DataFrame) -> np.ndarray:
    """
    Perhitungan probabilitas eksak menggunakan parameter terlatih dari models/churn_pipeline.pkl.
    Menghasilkan nilai probabilitas yang 100% identik dengan pipeline Logistic Regression asli.
    """
    X = preprocess(df)
    w = EXACT_MODEL_WEIGHTS
    c = w["coef"]
    tenure_log_scaled = (np.log1p(X["tenure"].astype(float)) - w["tenure_log_mean"]) / w["tenure_log_scale"]
    monthly_scaled = (X["MonthlyCharges"].astype(float) - w["monthly_mean"]) / w["monthly_scale"]

    z = (
        w["intercept"]
        + c[0] * (X["Dependents"] == "Yes").astype(float)
        + c[1] * (X["PaperlessBilling"] == "Yes").astype(float)
        + c[2] * (X["OnlineSecurity"] == "Yes").astype(float)
        + c[3] * (X["OnlineBackup"] == "Yes").astype(float)
        + c[4] * (X["DeviceProtection"] == "Yes").astype(float)
        + c[5] * (X["TechSupport"] == "Yes").astype(float)
        + c[6] * (X["InternetService"] == "Fiber optic").astype(float)
        + c[7] * (X["InternetService"] == "No").astype(float)
        + c[8] * (X["Contract"] == "One year").astype(float)
        + c[9] * (X["Contract"] == "Two year").astype(float)
        + c[10] * tenure_log_scaled
        + c[11] * monthly_scaled
    )
    return (1.0 / (1.0 + np.exp(-z))).to_numpy(dtype=float)


def model_predict_proba(model, df: pd.DataFrame) -> np.ndarray:
    """Kembalikan array 1D probabilitas kelas positif (Churn = Yes / 1)."""
    X = preprocess(df)
    try:
        return np.asarray(model.predict_proba(X))[:, 1]
    except Exception:
        return exact_logreg_predict_proba(X)


def predict_proba(df: pd.DataFrame) -> np.ndarray:
    """Fungsi prediksi utama yang dipakai di seluruh komponen aplikasi."""
    model, _ = load_model()
    if model is None:
        return exact_logreg_predict_proba(df)
    return model_predict_proba(model, df)


def get_model_status():
    model, path = load_model()
    return (model is not None), path


# =============================================================================
# DATA & VALIDASI
# =============================================================================
@st.cache_data(show_spinner=False)
def load_reference_data(path_or_buffer) -> pd.DataFrame:
    df = pd.read_csv(path_or_buffer)
    if ID_COL not in df.columns:
        df.insert(0, ID_COL, np.arange(len(df), dtype="int64"))
    return df


def get_reference_data() -> pd.DataFrame:
    for path in CANDIDATE_DATA_PATHS:
        if path.exists():
            return load_reference_data(path)
    st.sidebar.warning("File dataset referensi tidak ditemukan secara otomatis. Silakan unggah CSV:")
    up = st.sidebar.file_uploader(
        "Upload data referensi (data_cleaned.csv / data_telco_customer_churn.csv)",
        type="csv",
        key="ref_upload",
    )
    if up is None:
        st.info("Silakan unggah dataset referensi (`data_cleaned.csv` atau `data_telco_customer_churn.csv`) di sidebar untuk memulai.")
        st.stop()
    return load_reference_data(up)


def validate_columns(df: pd.DataFrame) -> list:
    return [c for c in FEATURE_COLS if c not in df.columns]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Samakan tipe data input dengan skema data training."""
    df = df.copy()
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CAT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("object").where(df[c].notna(), np.nan)
    return df


def parse_target_series(s: pd.Series) -> pd.Series:
    """Konversi kolom target Churn ('Yes'/'No' atau 1/0) menjadi integer 1/0."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(int)
    mapped = s.astype(str).str.strip().map({"Yes": 1, "No": 0, "1": 1, "0": 0, "True": 1, "False": 0})
    return mapped


def risk_label(p: np.ndarray, threshold: float) -> np.ndarray:
    return np.where(p >= threshold, "Churn / Berisiko (1)", "Tidak Churn / Bertahan (0)")


def compute_encoded_row(row_df: pd.DataFrame) -> pd.DataFrame:
    """Tampilkan representasi 12 fitur ter-encode yang masuk ke Logistic Regression."""
    X = preprocess(row_df)
    w = EXACT_MODEL_WEIGHTS
    return pd.DataFrame({
        "Dependents_Yes": (X["Dependents"] == "Yes").astype(int),
        "PaperlessBilling_Yes": (X["PaperlessBilling"] == "Yes").astype(int),
        "OnlineSecurity": (X["OnlineSecurity"] == "Yes").astype(int),
        "OnlineBackup": (X["OnlineBackup"] == "Yes").astype(int),
        "DeviceProtection": (X["DeviceProtection"] == "Yes").astype(int),
        "TechSupport": (X["TechSupport"] == "Yes").astype(int),
        "InternetService_Fiber optic": (X["InternetService"] == "Fiber optic").astype(int),
        "InternetService_No": (X["InternetService"] == "No").astype(int),
        "Contract_One year": (X["Contract"] == "One year").astype(int),
        "Contract_Two year": (X["Contract"] == "Two year").astype(int),
        "tenure_log": ((np.log1p(X["tenure"].astype(float)) - w["tenure_log_mean"]) / w["tenure_log_scale"]).round(4),
        "MonthlyCharges": ((X["MonthlyCharges"].astype(float) - w["monthly_mean"]) / w["monthly_scale"]).round(4),
    })


# =============================================================================
# MODEL BIAYA BISNIS (COST-BENEFIT ANALYSIS)
# =============================================================================
def business_outcome(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    monthly: np.ndarray,
    horizon: int = DEFAULT_HORIZON_MONTHS,
    success: float = DEFAULT_SUCCESS_RATE,
    offer_months: float = DEFAULT_OFFER_MONTHS,
) -> dict:
    """Perhitungan ekspektasi biaya retensi & pendapatan berisiko (sesuai src/evaluation.py)."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    monthly = np.asarray(monthly, dtype=float)

    revenue_horizon = horizon * monthly
    contacted = y_pred == 1
    churner = y_true == 1

    offer_spend = float((offer_months * monthly)[contacted].sum())
    revenue_at_risk = float(revenue_horizon[churner].sum())
    revenue_retained = float(success * revenue_horizon[churner & contacted].sum())
    revenue_lost = revenue_at_risk - revenue_retained

    return {
        "Pelanggan Dihubungi": int(contacted.sum()),
        "Churner Terjangkau": int((contacted & churner).sum()),
        "Biaya Penawaran (Offer Spend)": offer_spend,
        "Pendapatan Berisiko (Revenue at Risk)": revenue_at_risk,
        "Ekspektasi Pendapatan Tertahan": revenue_retained,
        "Ekspektasi Pendapatan Hilang": revenue_lost,
        "Total Ekspektasi Biaya": offer_spend + revenue_lost,
    }


# =============================================================================
# INTERPRETABILITAS (MODEL-AGNOSTIC & LOGREG COEFFICIENTS)
# =============================================================================
@st.cache_data(show_spinner=False)
def get_background(ref_df: pd.DataFrame, n: int = 150) -> pd.DataFrame:
    """Sampel data referensi sebagai baseline untuk penjelasan lokal (marginal replacement)."""
    return (
        ref_df[FEATURE_COLS]
        .sample(min(n, len(ref_df)), random_state=RANDOM_STATE)
        .reset_index(drop=True)
    )


def local_contributions(rows: pd.DataFrame, background: pd.DataFrame) -> pd.DataFrame:
    """
    Kontribusi tiap fitur terhadap prediksi (metode occlusion / marginal replacement):
        kontribusi_j = p(x) - rata-rata_b[ p(x dengan fitur j diganti nilai dari baris background b) ]
    Positif -> fitur tersebut MENAIKKAN risiko churn dibanding kondisi tipikal populasi
    Negatif -> fitur tersebut MENURUNKAN risiko churn dibanding kondisi tipikal populasi
    """
    rows = rows[FEATURE_COLS].reset_index(drop=True)
    n_rows, n_bg = len(rows), len(background)
    base_pred = predict_proba(rows)

    rep = rows.loc[rows.index.repeat(n_bg)].reset_index(drop=True)
    bg_tiled = pd.concat([background] * n_rows, ignore_index=True)

    contrib = {}
    for col in FEATURE_COLS:
        tmp = rep.copy()
        tmp[col] = bg_tiled[col].to_numpy()
        p = predict_proba(tmp).reshape(n_rows, n_bg).mean(axis=1)
        contrib[col] = base_pred - p
    return pd.DataFrame(contrib)


@st.cache_data(show_spinner=False)
def compute_permutation_importance(
    ref_df: pd.DataFrame,
    metric: str = "pr_auc",
    n_repeats: int = 5,
    _model_key: str = "default",
):
    """
    Global Permutation Importance: seberapa besar metrik (PR-AUC atau ROC-AUC) turun
    ketika satu fitur diacak secara independen.
    """
    if TARGET_COL not in ref_df.columns:
        return None, None
    y_series = parse_target_series(ref_df[TARGET_COL])
    valid_mask = y_series.notna()
    if valid_mask.sum() == 0:
        return None, None

    X = ref_df.loc[valid_mask, FEATURE_COLS].reset_index(drop=True)
    y = y_series.loc[valid_mask].astype(int).to_numpy()

    scorer = average_precision_score if metric == "pr_auc" else roc_auc_score
    base_score = float(scorer(y, predict_proba(X)))

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for col in FEATURE_COLS:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].to_numpy())
            drops.append(base_score - float(scorer(y, predict_proba(Xp))))
        rows.append({
            "feature": col,
            "importance": float(np.mean(drops)),
            "std": float(np.std(drops)),
        })
    imp = pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["label"] = imp["feature"].map(label)
    return imp, base_score


@st.cache_data(show_spinner=False)
def partial_dependence(
    ref_df: pd.DataFrame,
    feature: str,
    n_sample: int = 800,
    n_grid: int = 25,
    _model_key: str = "default",
) -> pd.DataFrame:
    """Rata-rata probabilitas churn ketika satu fitur di-set ke nilai tertentu (PDP)."""
    X = ref_df[FEATURE_COLS].sample(min(n_sample, len(ref_df)), random_state=RANDOM_STATE)
    if feature in CAT_COLS:
        grid = CATEGORY_ORDER.get(feature, ref_df[feature].dropna().unique().tolist())
    else:
        vals = ref_df[feature].dropna()
        grid = np.unique(np.quantile(vals, np.linspace(0.01, 0.99, n_grid)))
        if feature == "tenure":
            grid = np.unique(np.round(grid).astype(int))
    out = []
    for g in grid:
        Xg = X.copy()
        Xg[feature] = g
        if feature == "InternetService" and g == "No":
            for addon in ADDON_COLS:
                Xg[addon] = "No internet service"
        out.append({"value": g, "avg_proba": float(predict_proba(Xg).mean())})
    return pd.DataFrame(out)


@st.cache_data(show_spinner=False)
def compute_threshold_curve(
    ref_df: pd.DataFrame,
    horizon: int,
    success: float,
    offer_months: float,
) -> pd.DataFrame:
    """Evaluasi metrik klasifikasi dan ekspektasi biaya di sepanjang grid threshold 0.05 - 0.95."""
    y = parse_target_series(ref_df[TARGET_COL]).astype(int).to_numpy()
    proba = predict_proba(ref_df)
    monthly = ref_df["MonthlyCharges"].astype(float).to_numpy()
    thresholds = np.round(np.arange(0.05, 0.951, 0.01), 2)

    rows = []
    for t in thresholds:
        pred = (proba >= t).astype(int)
        cost_info = business_outcome(y, pred, monthly, horizon=horizon, success=success, offer_months=offer_months)
        rows.append({
            "threshold": float(t),
            "precision": float(precision_score(y, pred, zero_division=0)) if pred.sum() else np.nan,
            "recall": float(recall_score(y, pred, zero_division=0)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "flagged_pct": float(pred.mean() * 100),
            "total_cost": cost_info["Total Ekspektasi Biaya"],
        })
    return pd.DataFrame(rows)


def plot_contribution_bar(contrib_series: pd.Series, top_n: int = 10, title: str = "Kontribusi Fitur"):
    s = contrib_series.reindex(contrib_series.abs().sort_values(ascending=False).index)[:top_n]
    s = s.iloc[::-1]
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in s.values]
    fig = go.Figure(
        go.Bar(
            x=s.values,
            y=[label(i) for i in s.index],
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.2%}" for v in s.values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    lo, hi = float(min(s.min(), 0)), float(max(s.max(), 0))
    pad = (hi - lo) * 0.25 or 0.01
    fig.update_xaxes(range=[lo - (pad if lo < 0 else 0), hi + (pad if hi > 0 else 0)])
    fig.update_layout(
        title=title,
        xaxis_title="Perubahan probabilitas churn (poin %)",
        xaxis_tickformat=".1%",
        height=80 + 38 * len(s),
        margin=dict(l=10, r=50, t=50, b=40),
    )
    return fig


def plot_gauge(p: float, threshold: float):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number+delta",
            value=p * 100,
            delta={
                "reference": threshold * 100,
                "increasing": {"color": "#d62728"},
                "decreasing": {"color": "#2ca02c"},
                "valueformat": ".2f",
            },
            number={"suffix": "%", "valueformat": ".2f"},
            title={"text": f"Probabilitas Churn (Threshold = {threshold:.0%})"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#d62728" if p >= threshold else "#2ca02c"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "rgba(44, 160, 44, 0.15)"},
                    {"range": [threshold * 100, 100], "color": "rgba(214, 39, 40, 0.15)"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.8,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=270, margin=dict(l=20, r=20, t=55, b=10))
    return fig


# =============================================================================
# UI: SIDEBAR & INISIALISASI DATA REFERENSI
# =============================================================================
ref = get_reference_data()
missing_ref = validate_columns(ref)
if missing_ref:
    st.error(f"Data referensi tidak memiliki kolom wajib: {missing_ref}")
    st.stop()

ref = coerce_types(ref)
background = get_background(ref)
is_pickle_loaded, loaded_model_path = get_model_status()
model_key = loaded_model_path if is_pickle_loaded else "exact_logreg_weights"

st.sidebar.title("⚙️ Pengaturan Model & Bisnis")

threshold = st.sidebar.slider(
    "Threshold Klasifikasi Churn",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_THRESHOLD,
    step=0.01,
    help=(
        "Pelanggan dengan probabilitas >= threshold akan diklasifikasikan sebagai Churn / Berisiko (1) "
        "dan direkomendasikan untuk diberikan intervensi retensi. "
        "Threshold 0.27 dipilih dari minimisasi expected cost pada nested OOF (mendekati break-even c / (s*H) = 0.278)."
    ),
)

top_n = st.sidebar.slider(
    "Jumlah Fitur Ditampilkan",
    min_value=5,
    max_value=len(FEATURE_COLS),
    value=len(FEATURE_COLS),
)

st.sidebar.markdown("---")
st.sidebar.subheader("💰 Asumsi Model Biaya Retensi")
st.sidebar.caption("Parameter ilustratif untuk menghitung ekspektasi biaya & penghematan retensi:")
horizon_months = st.sidebar.number_input(
    "Horizon Pendapatan Hilang H (bulan)",
    min_value=1,
    max_value=60,
    value=DEFAULT_HORIZON_MONTHS,
    step=1,
    help="Jumlah bulan pendapatan yang hilang ketika seorang pelanggan churn (default proyek: 12 bulan).",
)
success_rate = st.sidebar.slider(
    "Tingkat Keberhasilan Retensi s",
    min_value=0.05,
    max_value=0.95,
    value=DEFAULT_SUCCESS_RATE,
    step=0.05,
    help="Proporsi pelanggan churn yang berhasil dipertahankan setelah diberi penawaran retensi (default: 30%).",
)
offer_cost_months = st.sidebar.number_input(
    "Biaya Penawaran Retensi c (bulan tagihan)",
    min_value=0.1,
    max_value=12.0,
    value=DEFAULT_OFFER_MONTHS,
    step=0.25,
    help="Biaya insentif/diskon retensi dalam satuan bulan MonthlyCharges pelanggan tersebut (default: 1.0 bulan).",
)
breakeven_t = offer_cost_months / (success_rate * horizon_months)
st.sidebar.info(f"📐 **Break-even Threshold Teoretis** ($c / (s \\cdot H)$): **{breakeven_t:.3f}**")

st.sidebar.markdown("---")
if is_pickle_loaded:
    st.sidebar.success(f"✅ **Model Pipeline Aktif** — `{Path(loaded_model_path).name}` (Logistic Regression, C=1)")
else:
    st.sidebar.info(
        "ℹ️ **Mode Bobot Eksak Aktif** — Menggunakan bobot terlatih Logistic Regression (`C=1`) "
        "dari `churn_pipeline.pkl` secara langsung."
    )

# =============================================================================
# UI: HEADER
# =============================================================================
st.title("📞 Telco Customer Churn Prediction")
st.caption(
    "Estimasi probabilitas churn pelanggan telekomunikasi berdasarkan 10 atribut profil, layanan, kontrak, dan biaya bulanan "
    "untuk mendukung keputusan intervensi retensi Tim Customer Success."
)

with st.expander("ℹ️ Ringkasan Proyek & Catatan Metodologi"):
    c_info1, c_info2, c_info3 = st.columns(3)
    with c_info1:
        st.markdown(
            "**Model Terpilih: Logistic Regression (`C=1`, L2)**\n"
            "- Dipilih melalui *5-fold Nested Cross-Validation* (PR-AUC: **0.666 ± 0.027**, ROC-AUC: **0.846**).\n"
            "- Performa pada *Internal Holdout Test Set* ($n=986$, $t=0.27$): Recall **81.0%**, Precision **51.6%**, "
            "F1 **63.0%**, ROC-AUC **0.854**, PR-AUC **0.685**, Brier **0.132**."
        )
    with c_info2:
        st.markdown(
            "**Preprocessing di Dalam Pipeline**\n"
            "- `Dependents`, `PaperlessBilling`: Binary One-Hot (`Yes=1`).\n"
            "- 4 Layanan Add-on: Binary Indicator (`Yes=1`, `No` & `No internet service` = `0`).\n"
            "- `InternetService`, `Contract`: One-Hot Encoding (referensi: `DSL` dan `Month-to-month`).\n"
            "- `tenure`: $\\log(1 + \\text{tenure})$ + StandardScaler; `MonthlyCharges`: StandardScaler."
        )
    with c_info3:
        st.markdown(
            "**Catatan Penting & Batasan**\n"
            "- Dataset bersifat *cross-sectional* historis tanpa tanggal observasi; probabilitas merepresentasikan "
            "estimasi risiko berdasarkan profil historis.\n"
            "- Artefak `models/churn_pipeline.pkl` telah di-*refit* pada seluruh 4.930 pelanggan berlabel."
        )

tab_single, tab_batch, tab_global = st.tabs(
    ["🧍 Single Prediction", "📂 Batch Prediction", "🔍 Interpretasi Model & Analisis Bisnis"]
)


# =============================================================================
# TAB 1: SINGLE PREDICTION
# =============================================================================
PRESET_PROFILES = {
    "Kustom / Input Manual": None,
    "Contoh Risiko Tinggi (README: Fiber optic, Month-to-month, Tenure 2 bln -> p = 82.4%)": {
        "Dependents": "No",
        "tenure": 2,
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "InternetService": "Fiber optic",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": 85.50,
    },
    "Contoh Risiko Rendah (README: DSL, Two year, Lengkap Add-on, Tenure 60 bln -> p = 1.0%)": {
        "Dependents": "Yes",
        "tenure": 60,
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "InternetService": "DSL",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "No",
        "MonthlyCharges": 70.00,
    },
}

with tab_single:
    st.subheader("Simulasi Prediksi Pelanggan Tunggal")

    preset_choice = st.selectbox(
        "Pilih Profil Preset (Opsional) untuk Mengisi Form Secara Cepat:",
        list(PRESET_PROFILES.keys()),
        index=1,
    )
    preset_vals = PRESET_PROFILES[preset_choice] or {
        "Dependents": "No",
        "tenure": int(ref["tenure"].median()),
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "InternetService": "Fiber optic",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "MonthlyCharges": float(ref["MonthlyCharges"].median()),
    }

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**1. Profil & Kontrak**")
            dependents = st.selectbox(
                label("Dependents"),
                CATEGORY_ORDER["Dependents"],
                index=CATEGORY_ORDER["Dependents"].index(preset_vals["Dependents"]),
                format_func=lambda x: "Ya (Yes)" if x == "Yes" else "Tidak (No)",
            )
            contract = st.selectbox(
                label("Contract"),
                CATEGORY_ORDER["Contract"],
                index=CATEGORY_ORDER["Contract"].index(preset_vals["Contract"]),
            )
            paperless = st.selectbox(
                label("PaperlessBilling"),
                CATEGORY_ORDER["PaperlessBilling"],
                index=CATEGORY_ORDER["PaperlessBilling"].index(preset_vals["PaperlessBilling"]),
                format_func=lambda x: "Ya (Yes)" if x == "Yes" else "Tidak (No)",
            )

        with c2:
            st.markdown("**2. Layanan Internet & Add-On**")
            internet_service = st.selectbox(
                label("InternetService"),
                CATEGORY_ORDER["InternetService"],
                index=CATEGORY_ORDER["InternetService"].index(preset_vals["InternetService"]),
            )
            online_security = st.selectbox(
                label("OnlineSecurity"),
                CATEGORY_ORDER["OnlineSecurity"],
                index=CATEGORY_ORDER["OnlineSecurity"].index(preset_vals["OnlineSecurity"]),
            )
            online_backup = st.selectbox(
                label("OnlineBackup"),
                CATEGORY_ORDER["OnlineBackup"],
                index=CATEGORY_ORDER["OnlineBackup"].index(preset_vals["OnlineBackup"]),
            )
            device_protection = st.selectbox(
                label("DeviceProtection"),
                CATEGORY_ORDER["DeviceProtection"],
                index=CATEGORY_ORDER["DeviceProtection"].index(preset_vals["DeviceProtection"]),
            )
            tech_support = st.selectbox(
                label("TechSupport"),
                CATEGORY_ORDER["TechSupport"],
                index=CATEGORY_ORDER["TechSupport"].index(preset_vals["TechSupport"]),
            )

        with c3:
            st.markdown("**3. Masa Berlangganan & Tagihan**")
            tenure = st.number_input(
                label("tenure"),
                min_value=0,
                max_value=120,
                value=int(preset_vals["tenure"]),
                step=1,
                help="Jumlah bulan pelanggan telah berlangganan (0 hingga 72+ bulan).",
            )
            monthly_charges = st.number_input(
                label("MonthlyCharges"),
                min_value=0.01,
                max_value=500.0,
                value=float(preset_vals["MonthlyCharges"]),
                step=0.50,
                format="%.2f",
                help="Biaya tagihan bulanan pelanggan (rentang dataset: 18.25 - 118.75).",
            )
            st.caption(
                "💡 *Catatan Struktural*: Jika `InternetService = No`, keempat layanan add-on otomatis "
                "disesuaikan menjadi `No internet service` sesuai aturan skema dataset."
            )

        submitted = st.form_submit_button("🔮 Prediksi Risiko Churn", type="primary", use_container_width=True)

    if submitted or "single_input" not in st.session_state:
        addons_adjusted = False
        raw_addons = {
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
        }
        if internet_service == "No":
            for k in raw_addons:
                if raw_addons[k] != "No internet service":
                    raw_addons[k] = "No internet service"
                    addons_adjusted = True
        else:
            for k in raw_addons:
                if raw_addons[k] == "No internet service":
                    raw_addons[k] = "No"
                    addons_adjusted = True

        input_df = pd.DataFrame([{
            "Dependents": dependents,
            "tenure": int(tenure),
            "OnlineSecurity": raw_addons["OnlineSecurity"],
            "OnlineBackup": raw_addons["OnlineBackup"],
            "InternetService": internet_service,
            "DeviceProtection": raw_addons["DeviceProtection"],
            "TechSupport": raw_addons["TechSupport"],
            "Contract": contract,
            "PaperlessBilling": paperless,
            "MonthlyCharges": float(monthly_charges),
        }])
        input_df = coerce_types(input_df)
        st.session_state["single_input"] = input_df
        st.session_state["addons_adjusted"] = addons_adjusted

    if "single_input" in st.session_state:
        input_df = st.session_state["single_input"]
        if st.session_state.get("addons_adjusted", False):
            st.info(
                "ℹ️ Nilai layanan add-on (`OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`) "
                "telah disesuaikan secara otomatis agar konsisten dengan status `InternetService`."
            )

        proba = float(predict_proba(input_df)[0])
        pred = risk_label(np.array([proba]), threshold)[0]
        avg_proba = float(predict_proba(background).mean())

        cust_monthly = float(input_df["MonthlyCharges"].iloc[0])
        rev_at_risk_if_churn = horizon_months * cust_monthly
        offer_cost_val = offer_cost_months * cust_monthly
        expected_saved_rev = proba * success_rate * rev_at_risk_if_churn
        net_expected_saving = expected_saved_rev - offer_cost_val

        st.markdown("---")
        r1, r2 = st.columns([1.05, 1.95])
        with r1:
            st.plotly_chart(plot_gauge(proba, threshold), use_container_width=True)
            if proba >= threshold:
                st.error(f"🚨 **Keputusan: {pred}** — Direkomendasikan untuk Kontak Retensi")
            else:
                st.success(f"✅ **Keputusan: {pred}** — Tidak Perlu Intervensi Saat Ini")

            m_c1, m_c2 = st.columns(2)
            m_c1.metric(
                "Probabilitas vs Rata-rata",
                f"{proba:.2%}",
                f"{proba - avg_proba:+.2%}",
                delta_color="inverse",
            )
            m_c2.metric(
                "Ekspektasi Net Saving Offer",
                f"${net_expected_saving:,.2f}",
                f"Biaya Offer: ${offer_cost_val:,.2f}",
                delta_color="normal" if net_expected_saving >= 0 else "inverse",
                help=(
                    f"Ekspektasi pendapatan yang diselamatkan (p * s * H * MonthlyCharges = ${expected_saved_rev:,.2f}) "
                    f"dikurangi biaya penawaran retensi (c * MonthlyCharges = ${offer_cost_val:,.2f})."
                ),
            )

        with r2:
            with st.spinner("Menghitung kontribusi tiap fitur..."):
                contrib = local_contributions(input_df, background).iloc[0]
            st.plotly_chart(
                plot_contribution_bar(contrib, top_n, "Faktor yang Paling Memengaruhi Prediksi Pelanggan Ini"),
                use_container_width=True,
            )
            st.caption(
                "🔴 **Merah** = menaikkan risiko churn  ·  🟢 **Hijau** = menurunkan risiko churn "
                "(dibandingkan profil pelanggan tipikal pada data referensi)."
            )

        up = contrib[contrib > 0].sort_values(ascending=False).head(4)
        down = contrib[contrib < 0].sort_values().head(4)
        e1, e2 = st.columns(2)
        e1.markdown(
            "**🔺 Faktor Penaik Risiko Churn Utama:**\n"
            + ("\n".join(f"- **{label(k)}** (`{input_df[k].iloc[0]}`): `{v:+.2%}`" for k, v in up.items()) or "- *(tidak ada)*")
        )
        e2.markdown(
            "**🔻 Faktor Penurun Risiko Churn Utama:**\n"
            + ("\n".join(f"- **{label(k)}** (`{input_df[k].iloc[0]}`): `{v:+.2%}`" for k, v in down.items()) or "- *(tidak ada)*")
        )

        with st.expander("🔎 Lihat Data Input Mentah & Representasi Fitur Ter-encode (Model Input)"):
            st.markdown("**Data Input Mentah (10 Fitur):**")
            st.dataframe(input_df, use_container_width=True)
            st.markdown("**Fitur Ter-encode & Ter-standarisasi (12 Fitur ke Logistic Regression):**")
            st.dataframe(compute_encoded_row(input_df), use_container_width=True)


# =============================================================================
# TAB 2: BATCH PREDICTION
# =============================================================================
with tab_batch:
    st.subheader("Prediksi Massal (Batch CSV) & Simulasi Kampanye Retensi")
    st.markdown(
        f"Unggah file CSV yang memiliki 10 kolom prediktor: `{', '.join(FEATURE_COLS)}`. "
        f"Kolom `{ID_COL}` dan `{TARGET_COL}` bersifat opsional (jika `{TARGET_COL}` tersedia, metrik evaluasi "
        f"dan analisis biaya aktual akan ditampilkan secara otomatis)."
    )

    col_dl1, col_dl2 = st.columns([1, 3])
    template_cols = [ID_COL] + FEATURE_COLS if ID_COL in ref.columns else FEATURE_COLS
    template_df = ref[template_cols].head(10)
    with col_dl1:
        st.download_button(
            "⬇️ Download Template CSV",
            template_df.to_csv(index=False).encode("utf-8"),
            "template_batch_telco_churn.csv",
            "text/csv",
            use_container_width=True,
        )
    with col_dl2:
        use_sample_data = st.checkbox(
            "Gunakan dataset referensi (`data_cleaned.csv` — 4.930 pelanggan) sebagai contoh batch sekarang",
            value=False,
        )

    batch_file = st.file_uploader("Upload File CSV untuk Batch Prediction", type="csv", key="batch_upload")

    batch = None
    if batch_file is not None:
        batch = pd.read_csv(batch_file)
    elif use_sample_data:
        batch = ref.copy()

    if batch is not None:
        missing = validate_columns(batch)
        if missing:
            st.error(f"Kolom wajib berikut tidak ditemukan di file CSV: {missing}")
            batch = None

    if batch is not None:
        if ID_COL not in batch.columns:
            batch.insert(0, ID_COL, np.arange(len(batch), dtype="int64"))
        batch = coerce_types(batch)

        n_bad_rows = int(batch[FEATURE_COLS].isna().any(axis=1).sum())
        if n_bad_rows > 0:
            st.warning(
                f"⚠️ Ditemukan {n_bad_rows} baris dengan nilai kosong/tidak valid. "
                "Baris dengan nilai kosong dihapus sebelum diprediksi."
            )
            batch = batch.dropna(subset=FEATURE_COLS).reset_index(drop=True)

        with st.spinner("Menjalankan prediksi model pada seluruh baris..."):
            proba = predict_proba(batch)

        result = batch.copy()
        result["churn_probability"] = np.round(proba, 4)
        result["flagged"] = (proba >= threshold).astype(int)
        result["label_prediksi"] = risk_label(proba, threshold)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Pelanggan", f"{len(result):,}")
        m2.metric("Diprediksi Churn (Flagged)", f"{int(result['flagged'].sum()):,}")
        m3.metric("% Pelanggan Di-flag", f"{result['flagged'].mean():.2%}")
        m4.metric("Rata-rata Probabilitas Churn", f"{proba.mean():.2%}")

        g1, g2 = st.columns(2)
        with g1:
            fig_hist = px.histogram(
                result,
                x="churn_probability",
                nbins=45,
                color="label_prediksi",
                color_discrete_map={
                    "Churn / Berisiko (1)": "#d62728",
                    "Tidak Churn / Bertahan (0)": "#2ca02c",
                },
                title="Distribusi Probabilitas Churn pada Batch",
                labels={"churn_probability": "Probabilitas Churn", "label_prediksi": "Status Prediksi"},
            )
            fig_hist.add_vline(
                x=threshold,
                line_dash="dash",
                line_color="black",
                annotation_text=f"Threshold = {threshold:.2f}",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        with g2:
            seg_col = st.selectbox("Rata-rata Risiko Churn per Segmen", CAT_COLS, format_func=label)
            seg = (
                result.assign(**{seg_col: result[seg_col].fillna("(Kosong)")})
                .groupby(seg_col, as_index=False)["churn_probability"]
                .mean()
                .sort_values("churn_probability")
            )
            fig_seg = px.bar(
                seg,
                x="churn_probability",
                y=seg_col,
                orientation="h",
                text_auto=".1%",
                title=f"Rata-rata Probabilitas Churn per {label(seg_col)}",
            )
            fig_seg.update_layout(xaxis_tickformat=".1%", yaxis_title=None, xaxis_title="Rata-rata Probabilitas Churn")
            st.plotly_chart(fig_seg, use_container_width=True)

        # Evaluasi & Analisis Biaya jika kolom target Churn tersedia
        if TARGET_COL in batch.columns:
            y_parsed = parse_target_series(batch[TARGET_COL])
            if y_parsed.notna().all() and y_parsed.nunique() == 2:
                st.markdown("---")
                st.markdown("#### 📏 Evaluasi Performa Model & Simulasi Biaya Retensi (Kolom `Churn` Ditemukan)")
                y = y_parsed.astype(int).to_numpy()
                pred_arr = result["flagged"].to_numpy()
                tn, fp, fn, tp = confusion_matrix(y, pred_arr, labels=[0, 1]).ravel()

                pr_auc_val = average_precision_score(y, proba)
                roc_auc_val = roc_auc_score(y, proba)
                brier_val = brier_score_loss(y, proba)
                prec_val = precision_score(y, pred_arr, zero_division=0)
                rec_val = recall_score(y, pred_arr, zero_division=0)
                f1_val = f1_score(y, pred_arr, zero_division=0)
                acc_val = accuracy_score(y, pred_arr)

                k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
                k1.metric("PR-AUC (Utama)", f"{pr_auc_val:.3f}")
                k2.metric("ROC-AUC (>=0.80)", f"{roc_auc_val:.3f}")
                k3.metric("Recall (>=0.80)", f"{rec_val:.3f}")
                k4.metric("Precision (>=0.60)", f"{prec_val:.3f}")
                k5.metric("F1-Score (>=0.65)", f"{f1_val:.3f}")
                k6.metric("Accuracy", f"{acc_val:.3f}")
                k7.metric("Brier Score", f"{brier_val:.3f}")

                cm_col, cost_col = st.columns([1, 1.6])
                with cm_col:
                    st.markdown("**Confusion Matrix (Threshold = " + f"`{threshold:.2f}`" + ")**")
                    cm_df = pd.DataFrame(
                        [[tn, fp], [fn, tp]],
                        index=["Aktual: Tidak Churn (0)", "Aktual: Churn (1)"],
                        columns=["Prediksi: Tidak Churn (0)", "Prediksi: Churn (1)"],
                    )
                    st.dataframe(cm_df, use_container_width=True)
                    st.caption(
                        f"**TP = {tp}** (Churner tertangkap) · **FP = {fp}** (False alarm) · "
                        f"**FN = {fn}** (Churner terlewat) · **TN = {tn}**"
                    )

                with cost_col:
                    st.markdown(
                        f"**Perbandingan Strategi Retensi ($H={horizon_months}$, $s={success_rate:.0%}$, $c={offer_cost_months}$)**"
                    )
                    monthly_arr = result["MonthlyCharges"].astype(float).to_numpy()
                    strat_dict = {
                        "Do nothing (Kontak 0)": np.zeros(len(y), dtype=int),
                        "Contact everyone (Kontak Semua)": np.ones(len(y), dtype=int),
                        f"Strategi Model (t = {threshold:.2f})": pred_arr,
                    }
                    cba_df = pd.DataFrame(
                        {
                            name: business_outcome(
                                y,
                                p_vec,
                                monthly_arr,
                                horizon=horizon_months,
                                success=success_rate,
                                offer_months=offer_cost_months,
                            )
                            for name, p_vec in strat_dict.items()
                        }
                    ).T
                    do_nothing_cost = cba_df.loc["Do nothing (Kontak 0)", "Total Ekspektasi Biaya"]
                    cba_df["Saving vs Do Nothing (%)"] = (
                        100.0 * (1.0 - cba_df["Total Ekspektasi Biaya"] / do_nothing_cost)
                    ).round(2)
                    st.dataframe(
                        cba_df[[
                            "Pelanggan Dihubungi",
                            "Churner Terjangkau",
                            "Biaya Penawaran (Offer Spend)",
                            "Ekspektasi Pendapatan Hilang",
                            "Total Ekspektasi Biaya",
                            "Saving vs Do Nothing (%)",
                        ]].style.format({
                            "Biaya Penawaran (Offer Spend)": "${:,.1f}",
                            "Ekspektasi Pendapatan Hilang": "${:,.1f}",
                            "Total Ekspektasi Biaya": "${:,.1f}",
                            "Saving vs Do Nothing (%)": "{:+.2f}%",
                        }),
                        use_container_width=True,
                    )

        st.markdown("#### 📋 Tabel Hasil Prediksi Batch")
        st.dataframe(
            result.sort_values("churn_probability", ascending=False),
            use_container_width=True,
            height=350,
        )
        st.download_button(
            "⬇️ Download Hasil Prediksi (CSV)",
            result.to_csv(index=False).encode("utf-8"),
            "hasil_prediksi_churn.csv",
            "text/csv",
            type="primary",
        )

        st.markdown("---")
        st.markdown("#### 🔍 Interpretasi Hasil Batch (Lokal & Global Batch)")
        b1, b2 = st.columns(2)
        with b1:
            idx_options = result.sort_values("churn_probability", ascending=False).index.tolist()[:200]
            show_id = ID_COL in result.columns
            row_idx = st.selectbox(
                "Pilih baris pelanggan untuk dijelaskan (diurutkan dari risiko tertinggi):",
                idx_options,
                format_func=lambda i: (
                    (f"{ID_COL}: {result.loc[i, ID_COL]}" if show_id else f"Baris {i}")
                    + f" — Probabilitas Churn: {result.loc[i, 'churn_probability']:.2%} ({result.loc[i, 'Contract']}, {result.loc[i, 'InternetService']})"
                ),
            )
            contrib_row = local_contributions(batch.loc[[row_idx]], background).iloc[0]
            st.plotly_chart(
                plot_contribution_bar(contrib_row, top_n, "Kontribusi Fitur (Pelanggan Terpilih)"),
                use_container_width=True,
            )

        with b2:
            max_rows = 250
            sample_b = batch.sample(min(max_rows, len(batch)), random_state=RANDOM_STATE)
            with st.spinner("Menghitung rata-rata pengaruh fitur di batch..."):
                bg_small = background.head(40)
                contrib_all = local_contributions(sample_b, bg_small)
            mean_abs = contrib_all.abs().mean()
            fig_batch_imp = plot_contribution_bar(mean_abs, top_n, "Rata-rata |Kontribusi| Fitur di Batch")
            fig_batch_imp.update_traces(
                marker_color="#1f77b4",
                text=[f"{v:.2%}" for v in mean_abs.sort_values(ascending=False)[:top_n].iloc[::-1]],
            )
            st.plotly_chart(fig_batch_imp, use_container_width=True)
            if len(batch) > max_rows:
                st.caption(f"Dihitung dari sampel acak {max_rows} baris agar responsif.")


# =============================================================================
# TAB 3: INTERPRETASI MODEL & ANALISIS BISNIS (GLOBAL)
# =============================================================================
with tab_global:
    st.subheader("1. Faktor Paling Berpengaruh Secara Global (Permutation Importance & Koefisien)")

    col_perm, col_coef = st.columns(2)

    with col_perm:
        st.markdown(
            "**Permutation Importance (Model-Agnostic)**\n\n"
            "Mengukur penurunan skor evaluasi ketika nilai satu fitur diacak. "
            "Pada proyek ini, **PR-AUC (Average Precision)** menjadi metrik utama."
        )
        imp_metric = st.radio(
            "Metrik Evaluasi Permutation Importance:",
            ["pr_auc", "roc_auc"],
            format_func=lambda m: "PR-AUC (Average Precision — Metrik Utama Proyek)" if m == "pr_auc" else "ROC-AUC",
            horizontal=True,
        )
        with st.spinner("Menghitung permutation importance..."):
            imp, base_score = compute_permutation_importance(ref, metric=imp_metric, _model_key=model_key)

        if imp is None:
            st.info(f"Data referensi tidak memiliki kolom `{TARGET_COL}`.")
        else:
            metric_name = "PR-AUC" if imp_metric == "pr_auc" else "ROC-AUC"
            st.metric(f"Baseline {metric_name} pada Data Referensi", f"{base_score:.3f}")
            top_imp = imp.head(top_n).iloc[::-1]
            fig_imp = go.Figure(
                go.Bar(
                    x=top_imp["importance"],
                    y=top_imp["label"],
                    orientation="h",
                    error_x=dict(type="data", array=top_imp["std"]),
                    marker_color="#4C72B0",
                    text=[f"{v:.3f}" for v in top_imp["importance"]],
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig_imp.update_layout(
                title=f"Permutation Importance (Penurunan {metric_name})",
                xaxis_title=f"Rata-rata Penurunan {metric_name}",
                height=100 + 36 * len(top_imp),
                margin=dict(l=10, r=45, t=50, b=40),
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    with col_coef:
        st.markdown(
            "**Koefisien & Odds Ratio Logistic Regression (`C = 1`)**\n\n"
            "Karena model final adalah Logistic Regression, setiap fitur ter-encode memiliki "
            "koefisien log-odds ($\\beta$) dan *Odds Ratio* ($e^\\beta$) yang dapat dibaca langsung."
        )
        coef_df = pd.DataFrame({
            "encoded_feature": EXACT_MODEL_WEIGHTS["encoded_features"],
            "label": [ENCODED_FEATURE_LABELS[f] for f in EXACT_MODEL_WEIGHTS["encoded_features"]],
            "coefficient": EXACT_MODEL_WEIGHTS["coef"],
        })
        coef_df["odds_ratio"] = np.exp(coef_df["coefficient"])
        coef_df = coef_df.sort_values("coefficient", ascending=True)

        st.metric("Intercept Model (Log-Odds Dasar)", f"{EXACT_MODEL_WEIGHTS['intercept']:.3f}")
        colors_coef = ["#DD8452" if c > 0 else "#4C72B0" for c in coef_df["coefficient"]]
        fig_coef = go.Figure(
            go.Bar(
                x=coef_df["coefficient"],
                y=coef_df["label"],
                orientation="h",
                marker_color=colors_coef,
                text=[f"{c:+.2f} (OR={or_v:.2f}x)" for c, or_v in zip(coef_df["coefficient"], coef_df["odds_ratio"])],
                textposition="outside",
                cliponaxis=False,
            )
        )
        fig_coef.update_xaxes(range=[-2.3, 1.1])
        fig_coef.update_layout(
            title="Koefisien Logistic Regression (Oranye = Naikkan Risiko, Biru = Turunkan)",
            xaxis_title="Koefisien (Perubahan Log-Odds Churn)",
            height=100 + 32 * len(coef_df),
            margin=dict(l=10, r=60, t=50, b=40),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")
    st.subheader("2. Bagaimana Satu Fitur Memengaruhi Prediksi? (Partial Dependence Plot)")
    st.markdown(
        "**Partial Dependence Plot (PDP)**: menampilkan rata-rata probabilitas churn ketika satu fitur "
        "di-set ke nilai tertentu pada seluruh populasi sampel, sementara fitur lainnya dipertahankan."
    )
    default_feat = "tenure" if "tenure" in FEATURE_COLS else FEATURE_COLS[0]
    pd_feat = st.selectbox(
        "Pilih Fitur untuk Partial Dependence Plot:",
        FEATURE_COLS,
        index=FEATURE_COLS.index(default_feat),
        format_func=label,
    )
    with st.spinner("Menghitung partial dependence..."):
        pdp = partial_dependence(ref, pd_feat, _model_key=model_key)

    if pd_feat in CAT_COLS:
        pdp = pdp.sort_values("avg_proba")
        fig_pdp = px.bar(
            pdp,
            x="avg_proba",
            y="value",
            orientation="h",
            text_auto=".1%",
            title=f"Partial Dependence: {label(pd_feat)}",
        )
        fig_pdp.update_layout(
            xaxis_tickformat=".1%",
            yaxis_title=label(pd_feat),
            xaxis_title="Rata-rata Probabilitas Churn",
        )
    else:
        fig_pdp = px.line(
            pdp,
            x="value",
            y="avg_proba",
            markers=True,
            title=f"Partial Dependence: {label(pd_feat)}",
        )
        fig_pdp.update_layout(
            yaxis_tickformat=".1%",
            xaxis_title=label(pd_feat),
            yaxis_title="Rata-rata Probabilitas Churn",
        )
    st.plotly_chart(fig_pdp, use_container_width=True)

    # Kurva Trade-off Threshold & Ekspektasi Biaya Bisnis
    if TARGET_COL in ref.columns:
        st.markdown("---")
        st.subheader("3. Analisis Pemilihan Threshold & Sensitivitas Biaya Retensi")
        st.markdown(
            "Menunjukkan trade-off antara **Precision**, **Recall**, dan **F1-Score**, serta kurva **Total Ekspektasi Biaya** "
            f"berdasarkan parameter bisnis di sidebar ($H={horizon_months}$, $s={success_rate:.0%}$, $c={offer_cost_months}$)."
        )
        tbl_thresh = compute_threshold_curve(ref, horizon_months, success_rate, offer_cost_months)
        opt_idx = tbl_thresh["total_cost"].idxmin()
        opt_t = float(tbl_thresh.loc[opt_idx, "threshold"])

        tc1, tc2 = st.columns(2)
        with tc1:
            fig_tm = go.Figure()
            for m_col, m_name, m_color in [
                ("precision", "Precision", "#1f77b4"),
                ("recall", "Recall", "#ff7f0e"),
                ("f1", "F1-Score", "#2ca02c"),
                ("accuracy", "Accuracy", "#d62728"),
            ]:
                fig_tm.add_trace(
                    go.Scatter(
                        x=tbl_thresh["threshold"],
                        y=tbl_thresh[m_col],
                        mode="lines",
                        name=m_name,
                        line=dict(color=m_color, width=2.5),
                    )
                )
            fig_tm.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Aktif ({threshold:.2f})")
            fig_tm.update_layout(
                title="Metrik Klasifikasi vs Threshold",
                xaxis_title="Classification Threshold",
                yaxis_title="Skor Metrik",
                yaxis_range=[0, 1.02],
                height=380,
            )
            st.plotly_chart(fig_tm, use_container_width=True)

        with tc2:
            fig_cost = go.Figure()
            fig_cost.add_trace(
                go.Scatter(
                    x=tbl_thresh["threshold"],
                    y=tbl_thresh["total_cost"] / 1e3,
                    mode="lines",
                    name="Strategi Model",
                    line=dict(color="#DD8452", width=3),
                )
            )
            fig_cost.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Aktif ({threshold:.2f})")
            fig_cost.add_vline(
                x=opt_t,
                line_dash="dot",
                line_color="#2ca02c",
                annotation_text=f"Min Cost ({opt_t:.2f})",
                annotation_position="bottom right",
            )
            fig_cost.update_layout(
                title=f"Total Ekspektasi Biaya vs Threshold (Minimum pada t = {opt_t:.2f})",
                xaxis_title="Classification Threshold",
                yaxis_title="Ekspektasi Biaya (Ribuan $)",
                height=380,
            )
            st.plotly_chart(fig_cost, use_container_width=True)