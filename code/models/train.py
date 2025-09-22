import argparse, os, json, joblib, pandas as pd, mlflow
from typing import Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import Pipeline

def evaluate(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    metrics = {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1)}
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/processed/train.csv")
    ap.add_argument("--test",  default="data/processed/test.csv")
    ap.add_argument("--out",   default="models/model.pkl")
    ap.add_argument("--metrics-out", default="models/metrics.json")
    args = ap.parse_args()

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("sms-spam")

    train = pd.read_csv(args.train)
    test  = pd.read_csv(args.test)

    Xtr, ytr = train["text"].astype(str).tolist(), train["y"].astype(int).values
    Xte, yte = test["text"].astype(str).tolist(),  test["y"].astype(int).values

    pipe = Pipeline(steps=[
        ("tfidf", TfidfVectorizer(strip_accents="unicode", lowercase=True, ngram_range=(1,2), min_df=2, max_df=0.95)),
        ("clf",  LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")),
    ])

    with mlflow.start_run():
        mlflow.autolog(log_input_examples=False, log_model_signatures=False)
        pipe.fit(Xtr, ytr)

        y_pred = pipe.predict(Xte)
        try:
            y_proba = pipe.predict_proba(Xte)[:, 1]
        except Exception:
            y_proba = None

        metrics = evaluate(yte, y_pred, y_proba)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        joblib.dump(pipe, args.out)

        os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        mlflow.log_artifact(args.out)
        mlflow.log_artifact(args.metrics_out)

        print("Training finished")
        print("Model:", args.out)
        print("Metrics:", json.dumps(metrics, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
