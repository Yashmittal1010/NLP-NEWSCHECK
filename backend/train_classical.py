import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
from app.text import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_NAMES = {0: "FAKE", 1: "REAL"}

def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    columns = {str(col).strip().lower(): col for col in df.columns}
    if "label" not in columns:
        raise ValueError(f"Expected a 'label' column. Found: {list(df.columns)}")
    text_columns = [columns[c] for c in ("title", "text") if c in columns]
    if not text_columns:
        raise ValueError("Expected at least one column named 'title' or 'text'.")
    text = df[text_columns].fillna("").astype(str).agg(" ".join, axis=1).map(normalize_text)
    labels = pd.to_numeric(df[columns["label"]], errors="coerce")
    labels = labels.map({0: 1, 1: 0})   # this CSV's raw convention is 0=real,1=fake — flip to match our 0=FAKE,1=REAL convention
    valid = labels.isin([0, 1]) & text.str.len().ge(10)
    clean = pd.DataFrame({"text": text[valid], "label": labels[valid].astype(int)})
    if set(clean["label"].unique()) != {0, 1}:
        raise ValueError("Expected both binary labels 0 and 1. Inspect the source dataset and label mapping.")
    print("ASSUMED label mapping: 0=FAKE, 1=REAL. Verify this against your dataset source.")
    print("Class counts:", {LABEL_NAMES[k]: int(v) for k, v in clean.label.value_counts().items()})
    conflicts = clean.groupby("text")["label"].nunique()
    conflicting_texts = set(conflicts[conflicts > 1].index)
    clean = clean[~clean.text.isin(conflicting_texts)].drop_duplicates("text")
    print("Rows after duplicate/conflict removal:", len(clean))
    return clean

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(PROJECT_ROOT / "data" / "raw" / "WELFake_Dataset.csv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    df = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        df.text, df.label, test_size=args.test_size, random_state=args.seed, stratify=df.label
    )
    vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 2),
                                 min_df=3, max_df=0.98, sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)
    models = {
        "logistic_regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "passive_aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=args.seed,
                                                          class_weight="balanced", early_stopping=True),
        "naive_bayes": MultinomialNB(alpha=0.1),
    }
    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "outputs").mkdir(exist_ok=True)
    reports = {}
    for name, model in models.items():
        model.fit(x_train_vec, y_train)
        pred = model.predict(x_test_vec)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
        reports[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
            "classification_report": classification_report(y_test, pred, labels=[0, 1],
                target_names=["FAKE", "REAL"], output_dict=True, zero_division=0),
            "confusion_matrix_fake_real": confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
        }
        print(f"{name}: accuracy={reports[name]['accuracy']:.4f}, macro_f1={reports[name]['macro_f1']:.4f}")
    joblib.dump({"vectorizer": vectorizer, "models": models, "label_mapping": LABEL_NAMES},
                PROJECT_ROOT / "models" / "classical_bundle.joblib")
    output = {
        "dataset": data_path.name, "rows_after_deduplication": len(df),
        "train_rows": len(x_train), "test_rows": len(x_test), "seed": args.seed,
        "label_mapping": LABEL_NAMES, "models": reports
    }
    (PROJECT_ROOT / "outputs" / "classical_metrics.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("Saved models/classical_bundle.joblib and outputs/classical_metrics.json")

if __name__ == "__main__":
    main()
