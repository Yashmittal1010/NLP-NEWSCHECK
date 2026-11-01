"""
diagnose_bundle.py

Diagnostic tool for NEWSCHECK's classical_bundle.joblib.

Place this file at the same level as train_classical.py
(i.e. backend/diagnose_bundle.py), so it can import `app.text`
the same way train_classical.py does.

Run from backend/ with the venv active:

    python diagnose_bundle.py

What it does:
  1. Loads the trained bundle (vectorizer + models + label_mapping).
  2. Prints the label_mapping actually stored in the bundle, so you can
     confirm 0=FAKE / 1=REAL is really what was trained (not just assumed).
  3. For Logistic Regression and Passive-Aggressive (both linear models
     with .coef_), prints the top N words/n-grams pushing predictions
     toward FAKE and toward REAL.
  4. For Naive Bayes, prints the top N words/n-grams by log-probability
     ratio between the two classes.
  5. Runs a small set of neutral, "clean prose" synthetic-style test
     sentences through all three models and prints label + confidence,
     so you can see directly whether clean/neutral register alone is
     enough to push predictions toward REAL.

This does not modify the bundle, the dataset, or any training code.
It only reads models/classical_bundle.joblib.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np

from app.text import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_PATH = PROJECT_ROOT / "models" / "classical_bundle.joblib"

TOP_N = 20

# Neutral, well-formed, "clean prose" test sentences with no obvious
# sensational language -- the kind of register a synthetic/LLM-written
# article tends to have regardless of whether its content is accurate.
TEST_SENTENCES = [
    "The city council approved a new budget plan on Tuesday after "
    "several hours of debate among local officials.",
    "Researchers at the university published a study this week "
    "examining long-term trends in regional rainfall patterns.",
    "The company announced quarterly earnings that exceeded analyst "
    "expectations, according to a statement released Thursday.",
    "A spokesperson confirmed that the proposed merger is still under "
    "review by regulators in both countries.",
    "Scientists have discovered a previously unknown species of frog "
    "living in a remote rainforest region, according to a new report.",
]


def print_top_features_linear(name, model, feature_names, label_mapping):
    if not hasattr(model, "coef_"):
        print(f"  [{name}] has no .coef_, skipping linear feature dump.")
        return

    coefs = model.coef_[0]
    order = np.argsort(coefs)

    fake_label = label_mapping.get(0, "0")
    real_label = label_mapping.get(1, "1")

    print(f"\n  Top {TOP_N} features pushing toward '{fake_label}' "
          f"(most negative coefficients):")
    for idx in order[:TOP_N]:
        print(f"    {feature_names[idx]:<30} {coefs[idx]:+.4f}")

    print(f"\n  Top {TOP_N} features pushing toward '{real_label}' "
          f"(most positive coefficients):")
    for idx in order[-TOP_N:][::-1]:
        print(f"    {feature_names[idx]:<30} {coefs[idx]:+.4f}")


def print_top_features_nb(name, model, feature_names, label_mapping):
    if not hasattr(model, "feature_log_prob_"):
        print(f"  [{name}] has no feature_log_prob_, skipping.")
        return

    # feature_log_prob_ shape: (n_classes, n_features), classes_ gives order
    classes = list(model.classes_)
    if len(classes) != 2:
        print(f"  [{name}] unexpected class count {classes}, skipping.")
        return

    idx0 = classes.index(0)
    idx1 = classes.index(1)
    log_ratio = model.feature_log_prob_[idx1] - model.feature_log_prob_[idx0]
    order = np.argsort(log_ratio)

    fake_label = label_mapping.get(0, "0")
    real_label = label_mapping.get(1, "1")

    print(f"\n  Top {TOP_N} features most associated with '{fake_label}' "
          f"(lowest log P(1)/P(0)):")
    for idx in order[:TOP_N]:
        print(f"    {feature_names[idx]:<30} {log_ratio[idx]:+.4f}")

    print(f"\n  Top {TOP_N} features most associated with '{real_label}' "
          f"(highest log P(1)/P(0)):")
    for idx in order[-TOP_N:][::-1]:
        print(f"    {feature_names[idx]:<30} {log_ratio[idx]:+.4f}")


def run_test_sentences(bundle, label_mapping):
    vectorizer = bundle["vectorizer"]
    models = bundle["models"]

    print("\n" + "=" * 70)
    print("TEST: neutral / clean-prose synthetic-style sentences")
    print("=" * 70)

    for sentence in TEST_SENTENCES:
        vec = vectorizer.transform([normalize_text(sentence)])
        print(f"\n> {sentence}")
        for name, model in models.items():
            pred = int(model.predict(vec)[0])
            label = label_mapping.get(pred, str(pred))
            conf = None
            if hasattr(model, "predict_proba"):
                conf = float(np.max(model.predict_proba(vec)[0]))
            conf_str = f"{conf * 100:.1f}%" if conf is not None else "n/a"
            print(f"    {name:<22} -> {label:<5} (confidence: {conf_str})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default=str(BUNDLE_PATH))
    args = parser.parse_args()

    bundle_path = Path(args.bundle)
    if not bundle_path.exists():
        raise SystemExit(
            f"Bundle not found at {bundle_path}. "
            f"Run train_classical.py first."
        )

    bundle = joblib.load(bundle_path)
    vectorizer = bundle["vectorizer"]
    models = bundle["models"]
    label_mapping = bundle.get("label_mapping", {0: "FAKE", 1: "REAL"})
    # joblib may round-trip int keys as-is; normalize just in case
    label_mapping = {int(k): v for k, v in label_mapping.items()}

    feature_names = vectorizer.get_feature_names_out()

    print("=" * 70)
    print("BUNDLE DIAGNOSTIC")
    print("=" * 70)
    print(f"Bundle path: {bundle_path}")
    print(f"Vocabulary size: {len(feature_names)}")
    print(f"Label mapping stored in bundle: {label_mapping}")
    print(f"Models in bundle: {list(models.keys())}")

    for name, model in models.items():
        print("\n" + "-" * 70)
        print(f"MODEL: {name} ({type(model).__name__})")
        print("-" * 70)
        if hasattr(model, "coef_"):
            print_top_features_linear(name, model, feature_names, label_mapping)
        elif hasattr(model, "feature_log_prob_"):
            print_top_features_nb(name, model, feature_names, label_mapping)
        else:
            print("  No inspectable linear/NB feature weights found.")

    run_test_sentences(bundle, label_mapping)

    print("\n" + "=" * 70)
    print("Done. Read the top-REAL feature lists above: if they're dominated")
    print("by generic/neutral words (the, said, according, officials, etc.)")
    print("rather than content-specific fake-news markers, that's consistent")
    print("with the model learning register/style rather than truthfulness.")
    print("=" * 70)


if __name__ == "__main__":
    main()