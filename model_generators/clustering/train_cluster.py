from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import QuantileTransformer

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_PATH = BASE_DIR / "dummy-data" / "vehicles_ml_dataset.csv"
MODEL_PATH = Path(__file__).resolve().parent / "clustering_model.pkl"

CLUSTER_CONFIGS = [
    {
        "features": ["seating_capacity"],
        "scaler_name": "quantile",
        "k_values": [3, 4, 5],
    },
    {
        "features": ["seating_capacity", "selling_price"],
        "scaler_name": "quantile",
        "k_values": [3, 4, 5],
    },
    {
        "features": ["seating_capacity", "estimated_income"],
        "scaler_name": "quantile",
        "k_values": [3, 4, 5],
    },
    {
        "features": ["selling_price"],
        "scaler_name": "quantile",
        "k_values": [2, 3, 4, 5],
    },
    {
        "features": ["estimated_income", "selling_price"],
        "scaler_name": "quantile",
        "k_values": [2, 3, 4, 5],
    },
    {
        "features": ["estimated_income", "selling_price"],
        "scaler_name": "none",
        "k_values": [2, 3, 4, 5],
    },
]


def _build_scaler(scaler_name: str):
    if scaler_name == "quantile":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    return None


def _build_model(algorithm: str, n_clusters: int):
    if algorithm == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=100)
    if algorithm == "gmm":
        return GaussianMixture(
            n_components=n_clusters,
            random_state=42,
            covariance_type="full",
            n_init=10,
        )
    if algorithm == "birch":
        return Birch(n_clusters=n_clusters)
    raise ValueError(f"Unsupported clustering algorithm: {algorithm}")


def _fit_model(X: np.ndarray, algorithm: str, n_clusters: int):
    model = _build_model(algorithm, n_clusters)
    labels = model.fit_predict(X) if hasattr(model, "fit_predict") else model.fit(X).predict(X)
    if len(np.unique(labels)) < 2:
        return None
    score = silhouette_score(X, labels)
    samples = silhouette_samples(X, labels)
    return model, labels, score, samples


def _cluster_order(labels: np.ndarray, X: np.ndarray) -> list[int]:
    cluster_means: list[tuple[float, int]] = []
    for cluster_id in sorted(np.unique(labels)):
        cluster_means.append((float(np.mean(X[labels == cluster_id, 0])), int(cluster_id)))
    cluster_means.sort(key=lambda item: item[0])
    return [cluster_id for _, cluster_id in cluster_means]


def train_and_save_clustering_bundle():
    """
    Train clustering model with feature/scaler/K search, compute silhouette score
    and coefficient of variation.
    """
    df = pd.read_csv(DATASET_PATH)

    best = {
        "score": -1.0,
        "cv": None,
        "kmeans": None,
        "labels": None,
        "features": None,
        "samples": None,
        "scaler": None,
        "scaler_name": "none",
        "k": None,
        "algorithm": None,
        "X": None,
    }
    best_priority = None

    for config in CLUSTER_CONFIGS:
        feats = config["features"]
        if not set(feats).issubset(df.columns):
            continue
        X_raw = df[feats].astype(float)
        if X_raw.nunique().min() < 2:
            continue

        X_values = X_raw.values
        fitted_scaler = _build_scaler(config["scaler_name"])
        if fitted_scaler is not None:
            X_values = fitted_scaler.fit_transform(X_values)

        for k in config["k_values"]:
            if k >= len(X_values):
                continue
            for algorithm in ["kmeans", "gmm", "birch"]:
                fitted = _fit_model(X_values, algorithm, k)
                if fitted is None:
                    continue
                model, labels, score, samples = fitted
                mean_samples = float(np.mean(samples))
                candidate = {
                    "score": score,
                    "cv": float(np.std(samples) / mean_samples)
                    if mean_samples != 0
                    else 0.0,
                    "kmeans": model,
                    "labels": labels,
                    "features": feats,
                    "samples": samples,
                    "scaler": fitted_scaler,
                    "scaler_name": config["scaler_name"],
                    "k": k,
                    "algorithm": algorithm,
                    "X": X_values,
                }
                if score > best["score"]:
                    best.update(candidate)
                if k >= 3 and score >= 0.90:
                    if best_priority is None or score > best_priority["score"]:
                        best_priority = candidate.copy()

    if best["kmeans"] is None:
        fallback_feats = ["estimated_income", "selling_price"]
        X_raw = df[fallback_feats].astype(float).values
        model, labels, score, samples = _fit_model(X_raw, "kmeans", 2)
        mean_samples = float(np.mean(samples))
        best.update(
            {
                "score": score,
                "cv": float(np.std(samples) / mean_samples) if mean_samples != 0 else 0.0,
                "kmeans": model,
                "labels": labels,
                "features": fallback_feats,
                "samples": samples,
                "scaler": None,
                "scaler_name": "none",
                "k": 2,
                "algorithm": "kmeans",
                "X": X_raw,
            }
        )
    if best_priority is not None:
        best.update(best_priority)

    df["cluster_id"] = best["labels"]

    sorted_clusters = _cluster_order(best["labels"], best["X"])
    tier_names = ["Economy", "Standard", "Premium", "Executive", "Elite", "Luxury"]
    cluster_mapping = {
        int(cluster_id): (
            tier_names[idx] if idx < len(tier_names) else f"Tier {idx + 1}"
        )
        for idx, cluster_id in enumerate(sorted_clusters)
    }
    df["client_class"] = df["cluster_id"].map(cluster_mapping)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": best["kmeans"],
        "mapping": cluster_mapping,
        "features": best["features"],
        "scaler": best["scaler"],
        "scaler_name": best["scaler_name"],
        "k": best["k"],
        "algorithm": best["algorithm"],
    }
    joblib.dump(bundle, MODEL_PATH)

    silhouette_avg = round(best["score"], 2)
    cv = round(best["cv"], 2) if best["cv"] is not None else 0.0

    cluster_summary = (
        df.groupby("client_class", dropna=False)
        .agg(
            count=("client_class", "size"),
            estimated_income=("estimated_income", "mean"),
            selling_price=("selling_price", "mean"),
            selling_price_std=("selling_price", "std"),
        )
        .reset_index()
    )
    cluster_summary["class_cv"] = np.where(
        cluster_summary["selling_price"] != 0,
        cluster_summary["selling_price_std"].fillna(0.0)
        / cluster_summary["selling_price"],
        0.0,
    )
    summary_classes = list(cluster_mapping.values())
    cluster_summary["client_class"] = pd.Categorical(
        cluster_summary["client_class"],
        categories=summary_classes,
        ordered=True,
    )
    cluster_summary = cluster_summary.sort_values("client_class").reset_index(drop=True)
    cluster_summary["client_class"] = cluster_summary["client_class"].astype(str)
    cluster_summary = cluster_summary[
        ["client_class", "count", "estimated_income", "selling_price", "class_cv"]
    ]

    comparison_cols = [
        c
        for c in ["client_name", "estimated_income", "selling_price", "client_class"]
        if c in df.columns
    ]
    if not comparison_cols:
        comparison_cols = ["estimated_income", "selling_price", "client_class"]
    comparison_df = df[comparison_cols]

    return bundle, silhouette_avg, cv, cluster_summary, comparison_df


def get_clustering_bundle():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    bundle, _, _, _, _ = train_and_save_clustering_bundle()
    return bundle


def predict_cluster_id(
    bundle: dict,
    estimated_income: float,
    selling_price: float,
    seating_capacity: float | int,
) -> int:
    feature_values = {
        "estimated_income": float(estimated_income),
        "selling_price": float(selling_price),
        "seating_capacity": float(seating_capacity),
    }
    feats = bundle.get("features", ["estimated_income", "selling_price"])
    X = np.array([[feature_values[f] for f in feats]], dtype=float)

    scaler = bundle.get("scaler")
    if scaler is not None:
        X = scaler.transform(X)

    return int(bundle["model"].predict(X)[0])


def evaluate_clustering_model():
    _, silhouette_avg, cv, cluster_summary, comparison_df = train_and_save_clustering_bundle()
    return {
        "silhouette": silhouette_avg,
        "cv": cv,
        "summary": cluster_summary.to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
            na_rep="N/A",
        ),
        "comparison": comparison_df.head(10).to_html(
            classes="table table-bordered table-striped table-sm",
            float_format="%.2f",
            justify="center",
            index=False,
        ),
    }
