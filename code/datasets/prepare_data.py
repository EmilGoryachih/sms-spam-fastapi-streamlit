import argparse, os, pandas as pd
from sklearn.model_selection import train_test_split

def load_raw(path: str) -> pd.DataFrame:
    df = None
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"):
        try:
            df = pd.read_csv(path, encoding=enc, engine="python", sep=None, on_bad_lines="skip")
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(f"Failed to read {path}. Last error: {last_err}")

    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if not {"label", "text"}.issubset(df.columns):
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "label", df.columns[1]: "text"})
        else:
            raise ValueError(f"Expected columns 'label' and 'text'. Got: {list(df.columns)}")

    df = df.dropna(subset=["label", "text"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["text"]  = df["text"].astype(str).str.strip()

    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)

    df["y"] = df["label"].apply(lambda s: 1 if "spam" in s else 0)
    return df[["text", "y"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw/spam.csv")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    df = load_raw(args.raw)
    X = df["text"]
    y = df["y"]

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    train = pd.DataFrame({"text": Xtr, "y": ytr})
    test  = pd.DataFrame({"text": Xte, "y": yte})

    train.to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    test.to_csv(os.path.join(args.outdir, "test.csv"), index=False)
    print("Saved:", os.path.join(args.outdir, "train.csv"), os.path.join(args.outdir, "test.csv"))

if __name__ == "__main__":
    main()
