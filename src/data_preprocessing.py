"""Data loading, validation, splitting and preprocessing for the Telco churn project.

Every learned transformation is returned inside a scikit-learn ``Pipeline`` / ``ColumnTransformer``
so that it is fitted only on the training portion of each split (no data leakage).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

# ----------------------------------------------------------------------------------- constants
RANDOM_STATE = 34
TEST_SIZE = 0.20
N_SPLITS = 5

TARGET = "Churn"
TARGET_MAP = {"No": 0, "Yes": 1}  # Churn = Yes is the positive class
RAW_DATA_RELATIVE = Path("data") / "raw" / "data_telco_customer_churn.csv"

FEATURES = ["Dependents", "tenure", "OnlineSecurity", "OnlineBackup", "InternetService",
            "DeviceProtection", "TechSupport", "Contract", "PaperlessBilling", "MonthlyCharges"]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
BINARY_FEATURES = ["Dependents", "PaperlessBilling"]
ADDON_FEATURES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
CATEGORICAL_FEATURES = [c for c in FEATURES if c not in NUMERIC_FEATURES]

CATEGORY_ORDER = {
    "Dependents": ["No", "Yes"],
    "PaperlessBilling": ["No", "Yes"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    **{c: ["No", "Yes", "No internet service"] for c in ADDON_FEATURES},
}
EXPECTED_CATEGORIES = {**CATEGORY_ORDER, TARGET: ["No", "Yes"]}
ADDON_ALLOWED = set(CATEGORY_ORDER["OnlineSecurity"])

TENURE_GROUP_BINS = [-1, 6, 12, 24, 48, 72]
TENURE_GROUP_LABELS = ["0-6", "7-12", "13-24", "25-48", "49-72"]

# Preprocessing configuration selected in notebook 00 (Sections 8-9) and used by every model notebook
FINAL_PREP_CONFIG = {"contract": "onehot", "addons": "binary", "tenure": "log"}


# ----------------------------------------------------------------------------------- data access
def find_project_root(start=None, marker: Path = RAW_DATA_RELATIVE) -> Path:
    """Walk up from ``start`` (default: current directory) until the raw dataset is found."""
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / marker).exists():
            return candidate
    raise FileNotFoundError(f"Could not locate {marker} from {start} or any parent directory.")


def load_raw_data(root=None) -> pd.DataFrame:
    """Read the raw CSV (read-only) and validate its schema."""
    root = find_project_root() if root is None else Path(root)
    df = pd.read_csv(root / RAW_DATA_RELATIVE)
    validate_data(df)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Fail loudly on missing columns, missing values, unexpected categories or invalid numbers."""
    missing_cols = set(FEATURES + [TARGET]) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing columns: {sorted(missing_cols)}")
    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Unexpected missing values in the data.")
    for col, allowed in EXPECTED_CATEGORIES.items():
        unexpected = set(df[col].unique()) - set(allowed)
        if unexpected:
            raise ValueError(f"Unexpected categories in {col}: {sorted(unexpected)}")
    if not pd.api.types.is_numeric_dtype(df["tenure"]) or not pd.api.types.is_numeric_dtype(df["MonthlyCharges"]):
        raise ValueError("tenure and MonthlyCharges must be numeric.")
    if (df["tenure"] < 0).any() or (df["MonthlyCharges"] <= 0).any():
        raise ValueError("Negative tenure or non-positive MonthlyCharges found.")


def split_features_target(df: pd.DataFrame):
    """Return X (the 10 predictors, original order) and y (Churn as 0/1)."""
    return df[FEATURES].copy(), df[TARGET].map(TARGET_MAP).astype(int)


def split_train_test(df: pd.DataFrame):
    """The project's single stratified 80/20 split (random_state = 34)."""
    X, y = split_features_target(df)
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


def get_cv() -> StratifiedKFold:
    """The shared 5-fold stratified splitter (identical folds for every experiment)."""
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


# ----------------------------------------------------------------------------------- encoders
def encode_addons(frame):
    """Add-on services: Yes -> 1, No / No internet service -> 0. Raises on unexpected values."""
    frame = pd.DataFrame(frame)
    unexpected = set(pd.unique(frame.to_numpy().ravel())) - ADDON_ALLOWED
    if unexpected:
        raise ValueError(f"Unexpected add-on service values: {unexpected}")
    return frame.eq("Yes").astype(int)


def count_addons(frame):
    """Number of add-on services subscribed (0-4)."""
    return encode_addons(frame).sum(axis=1).to_frame("n_addon_services")


def _count_name(transformer, input_features):
    return np.array(["n_addon_services"])


def _log_tenure_name(transformer, input_features):
    return np.array(["tenure_log"])


def make_preprocessor(features=None, contract="onehot", addons="binary", addon_count=False, tenure="raw",
                      extra_numeric=(), extra_binary=()):
    """Build the ColumnTransformer for a feature subset and encoding variant.

    contract : "onehot" (reference Month-to-month) or "ordinal" (0/1/2)
    addons   : "binary" (Yes=1, otherwise 0) or "onehot3" (keeps the structural 'No internet service' level)
    tenure   : "raw" (scaled months), "log" (scaled log1p months) or "group" (one-hot life-cycle bands)
    extra_numeric / extra_binary : engineered columns produced by feature_engineering.engineer_features
    """
    features = FEATURES if features is None else features
    keep = lambda cols: [c for c in cols if c in features]
    steps = []

    if keep(BINARY_FEATURES):
        cols = keep(BINARY_FEATURES)
        steps.append(("binary", OneHotEncoder(categories=[CATEGORY_ORDER[c] for c in cols], drop="first",
                                              sparse_output=False), cols))
    if keep(ADDON_FEATURES):
        cols = keep(ADDON_FEATURES)
        if addons == "binary":
            steps.append(("addons", FunctionTransformer(encode_addons, feature_names_out="one-to-one"), cols))
        elif addons == "onehot3":
            steps.append(("addons", OneHotEncoder(categories=[CATEGORY_ORDER[c] for c in cols], drop="first",
                                                  sparse_output=False), cols))
        else:
            raise ValueError(addons)
        if addon_count:
            steps.append(("addon_count", FunctionTransformer(count_addons, feature_names_out=_count_name), cols))
    if "InternetService" in features:
        steps.append(("internet", OneHotEncoder(categories=[CATEGORY_ORDER["InternetService"]], drop="first",
                                                sparse_output=False), ["InternetService"]))
    if "Contract" in features:
        if contract == "onehot":
            enc = OneHotEncoder(categories=[CATEGORY_ORDER["Contract"]], drop="first", sparse_output=False)
        elif contract == "ordinal":
            enc = OrdinalEncoder(categories=[CATEGORY_ORDER["Contract"]])
        else:
            raise ValueError(contract)
        steps.append(("contract", enc, ["Contract"]))

    numeric_cols = keep(NUMERIC_FEATURES)
    if "tenure" in numeric_cols and tenure != "raw":
        numeric_cols = [c for c in numeric_cols if c != "tenure"]
        if tenure == "log":
            steps.append(("tenure_log", Pipeline([
                ("log1p", FunctionTransformer(np.log1p, feature_names_out=_log_tenure_name)),
                ("scale", StandardScaler())]), ["tenure"]))
        elif tenure == "group":
            steps.append(("tenure_group", OneHotEncoder(categories=[TENURE_GROUP_LABELS], drop="first",
                                                        sparse_output=False), ["tenure_group"]))
        else:
            raise ValueError(tenure)
    if numeric_cols or extra_numeric:
        steps.append(("numeric", StandardScaler(), numeric_cols + list(extra_numeric)))
    if extra_binary:
        steps.append(("engineered_binary", "passthrough", list(extra_binary)))
    return ColumnTransformer(steps, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(model, engineer=False, sampler=None, **prep_kwargs):
    """Preprocessing (+ optional feature engineering / resampling) + model in one pipeline.

    engineer : prepend the stateless candidate-feature step (feature_engineering.engineer_features)
    sampler  : an imbalanced-learn sampler; it is placed first and is applied ONLY during fit,
               i.e. only to the training part of each fold - never to validation or test data.
    """
    steps = []
    if engineer:
        from src.feature_engineering import engineer_features  # local import avoids a circular import
        steps.append(("engineer", FunctionTransformer(engineer_features)))
    if sampler is not None:
        steps.append(("resample", sampler))
    steps += [("preprocess", make_preprocessor(**prep_kwargs)), ("model", clone(model))]
    if sampler is not None:
        from imblearn.pipeline import Pipeline as ImbPipeline
        return ImbPipeline(steps)
    return Pipeline(steps)


# ----------------------------------------------------------------------------------- fully nested check
def preprocessing_candidates() -> dict:
    """Every distinct preprocessing / feature-set configuration evaluated in notebook 00 (Sections 8-10).

    Used by notebook 08 to repeat those choices INSIDE the inner loop of nested CV. All candidates expect
    the stateless ``engineer`` step (``build_pipeline(..., engineer=True)``) so engineered columns exist.
    """
    base = dict(contract="onehot", addons="binary")
    configs = {
        "Encoding A (final encoding), raw tenure": dict(base),
        "Encoding B: ordinal Contract": dict(contract="ordinal", addons="binary"),
        "Encoding C: 3-level add-ons": dict(contract="onehot", addons="onehot3"),
        "A + tenure_log (FINAL_PREP_CONFIG)": dict(FINAL_PREP_CONFIG),
        "A + tenure_group": dict(base, tenure="group"),
        "A + mtm_fiber": dict(base, extra_binary=["mtm_fiber"]),
        "A + tenure_x_mtm": dict(base, extra_numeric=["tenure_x_mtm"]),
        "A + n_addon_services": dict(base, extra_numeric=["n_addon_services"]),
        "A + tenure_log + all interactions": dict(FINAL_PREP_CONFIG, extra_binary=["mtm_fiber"],
                                                  extra_numeric=["tenure_x_mtm", "n_addon_services"]),
    }
    for feature in FEATURES:
        configs[f"A + tenure_log without {feature}"] = dict(FINAL_PREP_CONFIG,
                                                            features=[f for f in FEATURES if f != feature])
    return {name: make_preprocessor(**kwargs) for name, kwargs in configs.items()}


# ----------------------------------------------------------------------------------- exported datasets
PROCESSED_DIR_RELATIVE = Path("data") / "processed"
ROW_ID = "row_id"  # 0-based row position in the raw CSV; identifies observations because exact duplicates exist
EXPORT_COLUMNS = [ROW_ID] + FEATURES + [TARGET]
EXPORT_FILES = {"cleaned": "data_cleaned.csv", "train": "data_train.csv", "test": "data_test.csv"}
SPLIT_METADATA_FILE = "split_metadata.json"
_STRING_COLUMNS = CATEGORICAL_FEATURES + [TARGET]
_EXPORT_DTYPES = {ROW_ID: "int64", "tenure": "int64", "MonthlyCharges": "float64",
                  **{c: "string" for c in _STRING_COLUMNS}}


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning decisions of notebook 00 (Section 5.7).

    The decisions are: keep all 77 exact duplicate rows, keep the 8 zero-tenure rows, remove no outliers,
    impute nothing (there are no missing values) and keep 'No internet service' as a structural category.
    The result therefore contains every raw observation, unchanged, with explicit data types and a
    ``row_id`` column. No encoding, scaling or feature selection is applied: those are learned inside
    the model pipeline on training folds only.
    """
    validate_data(df)
    if not df.index.equals(pd.RangeIndex(len(df))):
        raise ValueError("Expected the raw frame with its default RangeIndex (row positions).")
    out = df[FEATURES + [TARGET]].copy()
    out.insert(0, ROW_ID, np.arange(len(df), dtype="int64"))
    return out.astype(_EXPORT_DTYPES)[EXPORT_COLUMNS]


def load_processed(name: str, root=None) -> pd.DataFrame:
    """Load an exported file with the documented data types; ``name`` is 'cleaned', 'train' or 'test'."""
    root = find_project_root() if root is None else Path(root)
    frame = pd.read_csv(root / PROCESSED_DIR_RELATIVE / EXPORT_FILES[name], dtype=_EXPORT_DTYPES)
    validate_data(frame)
    return frame


def export_processed_datasets(root=None) -> dict:
    """Write data_cleaned.csv, data_train.csv and data_test.csv plus split_metadata.json.

    The split is NOT drawn again independently: row membership comes from ``split_train_test`` - the
    same call every notebook makes - and the train/test files keep its row order, so the shared CV
    folds (``get_cv``) are reproduced exactly when the exported training file is used.
    """
    import hashlib
    import json

    root = find_project_root() if root is None else Path(root)
    raw = load_raw_data(root)
    cleaned = clean_data(raw)
    X_train, X_test, _, _ = split_train_test(raw)
    frames = {"cleaned": cleaned, "train": cleaned.loc[X_train.index], "test": cleaned.loc[X_test.index]}
    out_dir = root / PROCESSED_DIR_RELATIVE
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {name: out_dir / EXPORT_FILES[name] for name in frames}
    for name, frame in frames.items():
        frame.to_csv(paths[name], index=False, lineterminator="\n")

    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    metadata = {
        "source": RAW_DATA_RELATIVE.as_posix(),
        "source_sha256": sha256(root / RAW_DATA_RELATIVE),
        "split_function": "src.data_preprocessing.split_train_test",
        "split": {"method": "sklearn.model_selection.train_test_split", "test_size": TEST_SIZE,
                  "stratify": TARGET, "shuffle": True, "random_state": RANDOM_STATE},
        "row_id": "0-based row position in the raw CSV",
        "row_order": "train/test rows are stored in split_train_test order (needed to reproduce the get_cv folds)",
        "files": {name: {"path": (PROCESSED_DIR_RELATIVE / EXPORT_FILES[name]).as_posix(), "rows": len(frame),
                         "columns": list(frame.columns), "churn_yes": int((frame[TARGET] == "Yes").sum()),
                         "sha256": sha256(paths[name])} for name, frame in frames.items()},
    }
    (out_dir / SPLIT_METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return paths


def verify_processed_datasets(root=None) -> pd.DataFrame:
    """Integrity checks on the exported files. Raises AssertionError on any failure; returns a summary table."""
    import json

    root = find_project_root() if root is None else Path(root)
    out_dir = root / PROCESSED_DIR_RELATIVE
    for file in [*EXPORT_FILES.values(), SPLIT_METADATA_FILE]:
        assert (out_dir / file).exists(), f"Missing {file}"                                   # 1. files exist
    frames = {name: load_processed(name, root) for name in EXPORT_FILES}                   # 9. pandas can load them
    cleaned, train, test = frames["cleaned"], frames["train"], frames["test"]
    raw = load_raw_data(root)
    X_train, X_test, y_train, y_test = split_train_test(raw)

    assert (len(cleaned), len(train), len(test)) == (len(raw), len(X_train), len(X_test))  # 2. row counts
    for frame in frames.values():
        assert list(frame.columns) == EXPORT_COLUMNS                                          # 3. columns and order
        assert int(frame.isna().sum().sum()) == 0                                             # 4. no missing values
    assert dict(train.dtypes) == dict(test.dtypes) == dict(cleaned.dtypes)                    #    identical dtypes
    assert set(train[ROW_ID]).isdisjoint(test[ROW_ID])                                        # 6. no overlap
    assert sorted([*train[ROW_ID], *test[ROW_ID]]) == cleaned[ROW_ID].tolist()                # 7. full coverage
    # The cleaned file reproduces the raw data exactly (no row removed, no value changed)
    pd.testing.assert_frame_equal(cleaned[FEATURES + [TARGET]], raw[FEATURES + [TARGET]], check_dtype=False)
    # Split membership AND row order are identical to the split used by every notebook
    for frame, X_ref, y_ref in [(train, X_train, y_train), (test, X_test, y_test)]:
        assert frame[ROW_ID].tolist() == X_ref.index.tolist()
        X_exp, y_exp = split_features_target(frame)
        pd.testing.assert_frame_equal(X_exp, X_ref.reset_index(drop=True), check_dtype=False)
        assert y_exp.tolist() == y_ref.tolist()
    metadata = json.loads((out_dir / SPLIT_METADATA_FILE).read_text(encoding="utf-8"))      # 8. split recorded
    assert metadata["split"]["random_state"] == RANDOM_STATE and metadata["split"]["test_size"] == TEST_SIZE

    rows = []
    for name, frame in frames.items():                                                      # 5. target distribution
        churn = frame[TARGET].eq("Yes")
        rows.append({"file": EXPORT_FILES[name], "rows": len(frame), "columns": frame.shape[1],
                     "share of cleaned rows (%)": 100 * len(frame) / len(cleaned),
                     "Churn = Yes": int(churn.sum()), "Churn = No": int((~churn).sum()),
                     "churn rate (%)": 100 * churn.mean(),
                     "exact duplicate rows (excl. row_id)": int(frame.drop(columns=ROW_ID).duplicated().sum()),
                     "missing values": int(frame.isna().sum().sum())})
    return pd.DataFrame(rows).set_index("file")
