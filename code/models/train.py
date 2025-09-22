import argparse
import json
import os
from typing import Dict

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")

    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if not {"label", "text"}.issubset(set(df.columns)):
        raise ValueError(
            f"Expected columns 'label' and 'text' in {path}. "
            f"Got columns: {list(df.columns)}"
        )

    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"] = df["text"].astype(str)

    def to01(x: str) -> int:
        xl = x.strip().lower()
        return 1 if "spam" in xl else 0

    df["y"] = df["label"].map(to01)
    return df[["text", "y"]]


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    strip_accents="unicode",
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../../data/raw/spam.csv", help="Path to raw CSV")
    ap.add_argument("--out", default="../../models/model.pkl", help="Where to save model")
    ap.add_argument(
        "--metrics-out", default="models/metrics.json", help="Where to save metrics"
    )
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = load_dataset(args.raw)
    X = df["text"].tolist()
    y = df["y"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None

    metrics = evaluate(y_test, y_pred, y_proba)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(pipe, args.out)

    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("Training finished.")
    print("Saved model to:", args.out)
    print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()