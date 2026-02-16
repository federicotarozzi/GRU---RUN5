"""
RUN5 – AIS Streaming GRU Classifier

This module implements:
- Ship-level split by MMSI (train/val/test)
- Window construction (lookback window, periodic updates, padding)
- Feature engineering (kinematic + temporal + basic vessel metadata + context information)
- GRU model (TensorFlow/Keras) with focal loss
- Threshold tuning on VAL at track-level with FAR constraint
- Streaming evaluation using cumulative-mean evidence

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import json
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class Config:
    # windowing
    lookback_hours: int = 1
    update_every_n_points: int = 10
    max_points_window: int = 100
    min_points_window: int = 20
    pad_value: float = -99.0

    # ship-level split
    random_seed: int = 42
    test_size_ships: float = 0.20
    val_size_ships: float = 0.20  # fraction of remaining ships after removing test

    # streaming evidence
    stream_consec_n: int = 2

    # threshold tuning (VAL constraint)
    far_target: float = 0.05
    eps_thr: float = 1e-6

    # feature engineering constants (seconds)
    gap_10min: float = 10 * 60
    gap_30min: float = 30 * 60
    max_dt_sec_for_div: float = 6 * 3600

    # training
    batch_size: int = 256
    epochs: int = 80
    lr: float = 1e-3
    patience_es: int = 10
    patience_rlr: int = 4


CFG = Config()


# ============================================================
# Utilities
# ============================================================

REQ_COLS = ["MMSI", "BaseDateTime", "LAT", "LON", "EVENT_ID", "LABEL"]


def parse_dt_series(s: pd.Series) -> pd.Series:
    s = s.astype("string")
    dt1 = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    m = dt1.isna()
    if m.any():
        dt2 = pd.to_datetime(s[m], format="%Y-%m-%dT%H:%M:%S", errors="coerce")
        dt1.loc[m] = dt2
    m2 = dt1.isna()
    if m2.any():
        dt1.loc[m2] = pd.to_datetime(s[m2], errors="coerce")
    return dt1


def label_to_int(x) -> Optional[int]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["g", "guilty", "1", "true", "t"]:
        return 1
    if s in ["n", "normal", "0", "false", "f"]:
        return 0
    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except Exception:
        return None


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def angdiff_deg(a, b) -> np.ndarray:
    return (a - b + 180.0) % 360.0 - 180.0


def norm_macro_from_vesseltype(v) -> str:
    if pd.isna(v):
        return "UNK"
    s = str(v).strip().upper()

    if s in ["CRUISE", "PASSENGER", "PAX"]:
        return "CRUISE"
    if s in ["TANKER", "TANK"]:
        return "TANKER"
    if s in ["CARGO", "BULK", "CONTAINER"]:
        return "CARGO"

    try:
        code = int(float(s))
        if 60 <= code <= 69:
            return "CRUISE"
        if 70 <= code <= 79:
            return "CARGO"
        if 80 <= code <= 89:
            return "TANKER"
    except Exception:
        pass

    return "UNK"


# ============================================================
# Feature list (model input)
# ============================================================

FEATURES: List[str] = [
    "LAT", "LON",
    "DT_SEC_RAW", "DT_SEC_LOG",
    "GAP_GT_10MIN", "GAP_GT_30MIN",
    "STEP_KM",
    "SOG_KMH", "GEO_KMH",
    "ACC_KMH_S",
    "COG_SIN", "COG_COS",
    "HDG_SIN", "HDG_COS",
    "TURN_DEG_S",
    "Length", "Width", "Draft",
    "VesselType", "Cargo", "Status",
    "IS_CRUISE", "IS_CARGO", "IS_TANKER",
]

# Contextual distances Related to specific SHP files
OPTIONAL_CONTEXT_COLS = ["DIST_LAND_KM", "DIST_PORTS_KM", "DIST_LANES_KM"]


# ============================================================
# Track-level feature engineering
# ============================================================

def add_track_features(track_df: pd.DataFrame) -> pd.DataFrame:
    out = track_df.copy()

    out["LAT"] = pd.to_numeric(out["LAT"], errors="coerce")
    out["LON"] = pd.to_numeric(out["LON"], errors="coerce")
    out = out.dropna(subset=["LAT", "LON", "BaseDateTime"]).copy()
    if out.empty:
        return out

    for c in ["SOG", "COG", "Heading", "Status", "Length", "Width", "Draft", "VesselType", "Cargo"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    out["DT_SEC_RAW"] = out["BaseDateTime"].diff().dt.total_seconds().fillna(0.0)
    out.loc[out["DT_SEC_RAW"] < 0, "DT_SEC_RAW"] = 0.0

    out["DT_SEC_LOG"] = np.log1p(out["DT_SEC_RAW"]).astype(np.float32)
    out["GAP_GT_10MIN"] = (out["DT_SEC_RAW"] > CFG.gap_10min).astype(np.float32)
    out["GAP_GT_30MIN"] = (out["DT_SEC_RAW"] > CFG.gap_30min).astype(np.float32)

    out["DT_SEC_DIV"] = (
        out["DT_SEC_RAW"].clip(lower=1.0, upper=float(CFG.max_dt_sec_for_div)).astype(np.float32)
    )

    lat_prev = out["LAT"].shift(1)
    lon_prev = out["LON"].shift(1)
    out["STEP_KM"] = haversine_km(lat_prev, lon_prev, out["LAT"], out["LON"]).fillna(0.0).astype(np.float32)

    sog = out["SOG"].fillna(0.0) if "SOG" in out.columns else 0.0
    out["SOG_KMH"] = (sog * 1.852).astype(np.float32)

    out["GEO_KMH"] = np.where(
        out["DT_SEC_DIV"] > 0,
        out["STEP_KM"] / (out["DT_SEC_DIV"] / 3600.0),
        0.0
    ).astype(np.float32)

    out["ACC_KMH_S"] = np.where(
        out["DT_SEC_DIV"] > 0,
        out["SOG_KMH"].diff().fillna(0.0) / out["DT_SEC_DIV"],
        0.0
    )
    out["ACC_KMH_S"] = out["ACC_KMH_S"].replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)

    if "COG" in out.columns:
        cog = (out["COG"].fillna(0.0) % 360.0).astype(float)
        out["COG_SIN"] = np.sin(np.radians(cog)).astype(np.float32)
        out["COG_COS"] = np.cos(np.radians(cog)).astype(np.float32)
        dcog = angdiff_deg(cog, cog.shift(1).fillna(cog))
        out["TURN_DEG_S"] = np.where(out["DT_SEC_DIV"] > 0, dcog / out["DT_SEC_DIV"], 0.0).astype(np.float32)
    else:
        out["COG_SIN"] = 0.0
        out["COG_COS"] = 0.0
        out["TURN_DEG_S"] = 0.0

    if "Heading" in out.columns:
        hdg = (out["Heading"].fillna(0.0) % 360.0).astype(float)
        out["HDG_SIN"] = np.sin(np.radians(hdg)).astype(np.float32)
        out["HDG_COS"] = np.cos(np.radians(hdg)).astype(np.float32)
    else:
        out["HDG_SIN"] = 0.0
        out["HDG_COS"] = 0.0

    if "VESSEL_MACRO" in out.columns:
        macro = str(out["VESSEL_MACRO"].iloc[0]).strip().upper()
    elif "VesselType" in out.columns:
        macro = norm_macro_from_vesseltype(out["VesselType"].iloc[0])
    else:
        macro = "UNK"

    out["IS_CRUISE"] = 1.0 if macro == "CRUISE" else 0.0
    out["IS_CARGO"]  = 1.0 if macro == "CARGO" else 0.0
    out["IS_TANKER"] = 1.0 if macro == "TANKER" else 0.0

    for c in ["Length", "Width", "Draft", "VesselType", "Cargo", "Status"]:
        if c in out.columns:
            out[c] = out[c].fillna(0.0)
        else:
            out[c] = 0.0

    # Context columns are treated as optional precomputed inputs
    for c in OPTIONAL_CONTEXT_COLS:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(np.float32)

    return out


def effective_feature_list(df: pd.DataFrame) -> List[str]:
    feats = FEATURES.copy()
    for c in OPTIONAL_CONTEXT_COLS:
        if c in df.columns:
            feats.append(c)
    return feats


# ============================================================
# Window construction
# ============================================================

def build_windows_for_track(track_df: pd.DataFrame, label: int, event_id: int, mmsi: int, feats: List[str]):
    rows = []
    n = len(track_df)
    if n < CFG.min_points_window:
        return rows

    idx_updates = list(range(0, n, CFG.update_every_n_points))
    if idx_updates and idx_updates[-1] != n - 1:
        idx_updates.append(n - 1)

    for i in idx_updates:
        end_time = track_df["BaseDateTime"].iloc[i]
        start_time = end_time - pd.Timedelta(hours=CFG.lookback_hours)

        sub = track_df[(track_df["BaseDateTime"] >= start_time) & (track_df["BaseDateTime"] <= end_time)].copy()
        if len(sub) < CFG.min_points_window:
            continue

        sub = sub.tail(CFG.max_points_window).copy()
        x = sub[feats].to_numpy(dtype=np.float32)

        if x.shape[0] < CFG.max_points_window:
            pad_len = CFG.max_points_window - x.shape[0]
            pad = np.full((pad_len, x.shape[1]), CFG.pad_value, dtype=np.float32)
            x = np.vstack([pad, x])

        meta = {
            "EVENT_ID": int(event_id),
            "MMSI": int(mmsi),
            "END_TIME": str(end_time),
            "N_POINTS_IN_WINDOW": int(len(sub)),
        }
        rows.append((x, int(label), meta))

    return rows


def build_samples(df_all: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    x_list, y_list, meta_list = [], [], []
    group_cols = ["EVENT_ID", "MMSI", "LABEL"]

    feats = effective_feature_list(df_all)

    gb = df_all.groupby(group_cols)
    for (event_id, mmsi, lab), tr in gb:
        tr = tr.sort_values("BaseDateTime").copy()
        tr = add_track_features(tr)
        if tr.empty:
            continue

        rows = build_windows_for_track(tr, label=int(lab), event_id=int(event_id), mmsi=int(mmsi), feats=feats)
        for xw, yw, meta in rows:
            x_list.append(xw)
            y_list.append(yw)
            meta_list.append(meta)

    if len(x_list) == 0:
        raise RuntimeError("No samples built. Check datetime parsing and min window constraints.")

    x = np.stack(x_list).astype(np.float32)
    y = np.array(y_list).astype(np.int32)
    meta_df = pd.DataFrame(meta_list)
    return x, y, meta_df, feats


# ============================================================
# Scaling (ignore padded rows)
# ============================================================

def compute_mean_std_ignore_pad(x3d: np.ndarray, pad_value: float) -> Tuple[np.ndarray, np.ndarray]:
    mask = (x3d[:, :, 0] != pad_value)  # LAT is used to detect padded rows
    n_features = x3d.shape[2]
    mean = np.zeros(n_features, dtype=np.float32)
    std = np.ones(n_features, dtype=np.float32)

    for j in range(n_features):
        vals = x3d[:, :, j][mask].astype(np.float32)
        if len(vals) == 0:
            mean[j] = 0.0
            std[j] = 1.0
        else:
            mean[j] = float(np.mean(vals))
            s = float(np.std(vals))
            std[j] = s if s > 1e-6 else 1.0
    return mean, std


def apply_mean_std_ignore_pad(x3d: np.ndarray, mean: np.ndarray, std: np.ndarray, pad_value: float) -> np.ndarray:
    xo = x3d.copy()
    mask = (xo[:, :, 0] != pad_value)
    for j in range(xo.shape[2]):
        v = xo[:, :, j]
        v_masked = v[mask]
        v[mask] = (v_masked - mean[j]) / std[j]
        xo[:, :, j] = v
    return xo


def scale_and_pad_window(feat_mat: np.ndarray, mean: np.ndarray, std: np.ndarray, pad_value: float, max_points: int) -> np.ndarray:
    x = feat_mat.astype(np.float32)
    if x.shape[0] < max_points:
        pad = np.full((max_points - x.shape[0], x.shape[1]), pad_value, dtype=np.float32)
        x = np.vstack([pad, x])

    mask = (x[:, 0] != pad_value)
    for j in range(x.shape[1]):
        v = x[:, j]
        v_masked = v[mask]
        v[mask] = (v_masked - mean[j]) / std[j]
        x[:, j] = v

    return x[None, ...]


# ============================================================
# Model
# ============================================================

def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    def _loss(y_true, y_pred):
        y_true_f = tf.cast(y_true, tf.float32)
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        p_t = y_true_f * y_pred + (1.0 - y_true_f) * (1.0 - y_pred)
        alpha_t = y_true_f * alpha + (1.0 - y_true_f) * (1.0 - alpha)
        loss = -alpha_t * tf.pow(1.0 - p_t, gamma) * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return _loss


def build_stream_gru(input_shape: Tuple[int, int], pad_value: float) -> tf.keras.Model:
    inp = layers.Input(shape=input_shape)
    x = layers.Masking(mask_value=pad_value)(inp)
    x = layers.GRU(64, return_sequences=True)(x)
    x = layers.Dropout(0.25)(x)
    x = layers.GRU(32, return_sequences=False)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    out = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CFG.lr),
        loss=focal_loss(gamma=2.0, alpha=0.25),
        metrics=[
            tf.keras.metrics.AUC(curve="PR", name="auc_pr"),
            tf.keras.metrics.AUC(curve="ROC", name="auc_roc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )
    return model


# ============================================================
# Evaluation utilities
# ============================================================

def eval_window_level(model: tf.keras.Model, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = model.predict(x, verbose=0).reshape(-1)
    yb = (p >= 0.5).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "auc_pr": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "acc": float(np.mean(yb == y)),
    }


def stream_scores_cummean(
    model: tf.keras.Model,
    track_raw: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    feats: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    tr = track_raw.sort_values("BaseDateTime").copy()
    tr = add_track_features(tr)
    if tr.empty or len(tr) < CFG.min_points_window:
        return None

    probs: List[float] = []
    scores: List[float] = []
    times: List[np.datetime64] = []

    idx_updates = list(range(0, len(tr), CFG.update_every_n_points))
    if idx_updates and idx_updates[-1] != len(tr) - 1:
        idx_updates.append(len(tr) - 1)

    for i in idx_updates:
        end_time = tr["BaseDateTime"].iloc[i]
        start_time = end_time - pd.Timedelta(hours=CFG.lookback_hours)

        sub = tr[(tr["BaseDateTime"] >= start_time) & (tr["BaseDateTime"] <= end_time)].copy()
        if len(sub) < CFG.min_points_window:
            continue

        sub = sub.tail(CFG.max_points_window).copy()
        feat = sub[feats].to_numpy(dtype=np.float32)
        x_in = scale_and_pad_window(feat, mean, std, pad_value=CFG.pad_value, max_points=CFG.max_points_window)

        p = float(model.predict(x_in, verbose=0)[0, 0])
        probs.append(p)
        times.append(np.datetime64(end_time))
        scores.append(float(np.mean(probs)))  # cumulative mean evidence

    if len(scores) == 0:
        return None

    return {
        "probs": np.array(probs, dtype=float),
        "scores": np.array(scores, dtype=float),
        "times": np.array(times, dtype="datetime64[ns]"),
        "t0": np.datetime64(tr["BaseDateTime"].iloc[0]),
    }


def compute_val_track_maxscores(
    model: tf.keras.Model,
    val_df: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    feats: List[str],
) -> pd.DataFrame:
    group_cols = ["EVENT_ID", "MMSI", "LABEL"]
    rows = []
    for (event_id, mmsi, lab), tr in val_df.groupby(group_cols):
        out = stream_scores_cummean(model, tr, mean, std, feats)
        if out is None:
            continue
        rows.append({
            "EVENT_ID": int(event_id),
            "MMSI": int(mmsi),
            "LABEL": int(lab),
            "MAX_SCORE": float(np.max(out["scores"])),
        })
    return pd.DataFrame(rows)


def tune_threshold_maximize_recall_under_far(
    val_track_scores: pd.DataFrame,
    far_target: float,
    eps: float,
) -> Tuple[float, Dict[str, float]]:
    normal = val_track_scores[val_track_scores["LABEL"] == 0]["MAX_SCORE"].to_numpy(dtype=float)
    guilty = val_track_scores[val_track_scores["LABEL"] == 1]["MAX_SCORE"].to_numpy(dtype=float)

    if len(normal) == 0 or len(guilty) == 0:
        raise RuntimeError("VAL must contain both normal and guilty tracks to tune threshold.")

    candidates = np.unique(np.concatenate([normal, guilty]))
    candidates = np.sort(candidates)

    best = None
    for thr in candidates:
        far = float(np.mean(normal >= thr))
        if far > far_target:
            continue
        rec = float(np.mean(guilty >= thr))
        key = (rec, -far, thr)
        if best is None or key > best["key"]:
            best = {"thr": float(thr + eps), "rec": rec, "far": far, "key": key}

    if best is None:
        thr_q = float(np.quantile(normal, 1.0 - far_target) + eps)
        best = {"thr": thr_q, "rec": float(np.mean(guilty >= thr_q)), "far": float(np.mean(normal >= thr_q)), "key": None}

    debug = {
        "n_val_normal_tracks": int(len(normal)),
        "n_val_guilty_tracks": int(len(guilty)),
        "far_target": float(far_target),
        "thr_selected": float(best["thr"]),
        "val_far_at_thr": float(best["far"]),
        "val_recall_at_thr": float(best["rec"]),
        "normal_p95": float(np.quantile(normal, 0.95)),
        "normal_p99": float(np.quantile(normal, 0.99)),
        "normal_max": float(np.max(normal)),
        "guilty_max": float(np.max(guilty)),
    }
    return best["thr"], debug


def track_report_streaming(
    model: tf.keras.Model,
    df_split: pd.DataFrame,
    mean: np.ndarray,
    std: np.ndarray,
    feats: List[str],
    thr: float,
    consec_n: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    group_cols = ["EVENT_ID", "MMSI", "LABEL"]
    recs = []

    for (event_id, mmsi, lab), tr in df_split.groupby(group_cols):
        out = stream_scores_cummean(model, tr, mean, std, feats)
        if out is None:
            continue

        scores = out["scores"]
        alerts = (scores >= thr).astype(int)

        consec = 0
        fired = False
        for a in alerts:
            consec = consec + 1 if a == 1 else 0
            if consec >= consec_n:
                fired = True
                break

        recs.append({
            "EVENT_ID": int(event_id),
            "MMSI": int(mmsi),
            "LABEL": int(lab),
            "MAX_SCORE": float(np.max(scores)),
            "ALERT": int(fired),
        })

    rep = pd.DataFrame(recs)
    if rep.empty:
        return rep, {"far": float("nan"), "recall": float("nan")}

    y_true = rep["LABEL"].to_numpy(dtype=int)
    y_pred = rep["ALERT"].to_numpy(dtype=int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    metrics = {"far": float(far), "recall": float(recall), "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)}
    return rep, metrics


# ============================================================
# Data loading and ship-level split
# ============================================================

def ship_level_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(CFG.random_seed)

    ships_g = np.unique(df[df["LABEL"] == 1]["MMSI"].to_numpy(dtype=np.int64))
    ships_n = np.unique(df[df["LABEL"] == 0]["MMSI"].to_numpy(dtype=np.int64))

    rng.shuffle(ships_g)
    rng.shuffle(ships_n)

    def split_ids(ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_total = len(ids)
        n_test = int(round(CFG.test_size_ships * n_total))
        test = ids[:n_test]
        remain = ids[n_test:]
        n_val = int(round(CFG.val_size_ships * len(remain)))
        val = remain[:n_val]
        train = remain[n_val:]
        return train, val, test

    g_train, g_val, g_test = split_ids(ships_g)
    n_train, n_val, n_test = split_ids(ships_n)

    train_ids = set(g_train.tolist() + n_train.tolist())
    val_ids = set(g_val.tolist() + n_val.tolist())
    test_ids = set(g_test.tolist() + n_test.tolist())

    train_df = df[df["MMSI"].isin(train_ids)].copy()
    val_df = df[df["MMSI"].isin(val_ids)].copy()
    test_df = df[df["MMSI"].isin(test_ids)].copy()
    return train_df, val_df, test_df


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=True)

    for c in REQ_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    df["LABEL"] = df["LABEL"].map(label_to_int)
    df = df.dropna(subset=["LABEL"]).copy()
    df["LABEL"] = df["LABEL"].astype("int64")

    df["MMSI"] = pd.to_numeric(df["MMSI"], errors="coerce")
    df["EVENT_ID"] = pd.to_numeric(df["EVENT_ID"], errors="coerce")
    df["BaseDateTime"] = parse_dt_series(df["BaseDateTime"])

    df = df.dropna(subset=["MMSI", "EVENT_ID", "BaseDateTime"]).copy()
    df["MMSI"] = df["MMSI"].astype("int64")
    df["EVENT_ID"] = df["EVENT_ID"].astype("int64")

    return df


# ============================================================
# Main routine
# ============================================================

def main(csv_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_and_clean(Path(csv_path))

    train_df, val_df, test_df = ship_level_split(df)

    x_train, y_train, meta_train, feats = build_samples(train_df)
    x_val, y_val, meta_val, _ = build_samples(val_df)
    x_test, y_test, meta_test, _ = build_samples(test_df)

    mean, std = compute_mean_std_ignore_pad(x_train, pad_value=CFG.pad_value)
    x_train_s = apply_mean_std_ignore_pad(x_train, mean, std, pad_value=CFG.pad_value)
    x_val_s = apply_mean_std_ignore_pad(x_val, mean, std, pad_value=CFG.pad_value)
    x_test_s = apply_mean_std_ignore_pad(x_test, mean, std, pad_value=CFG.pad_value)

    with open(out / "scaler_mean_std.json", "w") as f:
        json.dump({"features": feats, "pad_value": CFG.pad_value, "mean": mean.tolist(), "std": std.tolist()}, f, indent=2)

    model = build_stream_gru((CFG.max_points_window, len(feats)), pad_value=CFG.pad_value)

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=CFG.patience_es, restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=CFG.patience_rlr, verbose=0),
        callbacks.ModelCheckpoint(str(out / "best_model.keras"), monitor="val_loss", save_best_only=True, verbose=0),
    ]

    model.fit(
        x_train_s, y_train,
        validation_data=(x_val_s, y_val),
        epochs=CFG.epochs,
        batch_size=CFG.batch_size,
        callbacks=cbs,
        verbose=0,
    )

    win_val = eval_window_level(model, x_val_s, y_val)
    win_test = eval_window_level(model, x_test_s, y_test)

    val_track_scores = compute_val_track_maxscores(model, val_df, mean, std, feats)
    thr, thr_debug = tune_threshold_maximize_recall_under_far(val_track_scores, far_target=CFG.far_target, eps=CFG.eps_thr)

    val_rep, val_metrics = track_report_streaming(model, val_df, mean, std, feats, thr=thr, consec_n=CFG.stream_consec_n)
    test_rep, test_metrics = track_report_streaming(model, test_df, mean, std, feats, thr=thr, consec_n=CFG.stream_consec_n)

    val_rep.to_csv(out / "val_track_level_streaming.csv", index=False)
    test_rep.to_csv(out / "test_track_level_streaming.csv", index=False)

    meta = {
        "inputs": {"csv_path": str(csv_path)},
        "config": CFG.__dict__,
        "features": feats,
        "thresholding": {"thr": float(thr), "stream_consec_n": int(CFG.stream_consec_n), "tuning_debug": thr_debug},
        "window_level_metrics": {"val": win_val, "test": win_test},
        "track_level_metrics": {"val": val_metrics, "test": test_metrics},
        "outputs": {
            "model": str(out / "best_model.keras"),
            "scaler": str(out / "scaler_mean_std.json"),
            "val_track_report": str(out / "val_track_level_streaming.csv"),
            "test_track_report": str(out / "test_track_level_streaming.csv"),
        },
    }
    with open(out / "run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="RUN5 reference implementation")
    p.add_argument("--csv", required=True, help="Path to AIS merged labeled CSV")
    p.add_argument("--out", required=True, help="Output directory")
    args = p.parse_args()
    main(args.csv, args.out)
