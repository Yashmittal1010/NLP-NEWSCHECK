from pathlib import Path
import joblib
import numpy as np
from .text import normalize_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_ROOT / "models" / "classical_bundle.joblib"
_bundle = None

def load_bundle():
    global _bundle
    if _bundle is not None:
        return _bundle
    if not BUNDLE_PATH.exists():
        return None
    _bundle = joblib.load(BUNDLE_PATH)
    return _bundle

def predict_one(text: str, model_name: str) -> dict:
    bundle = load_bundle()
    if bundle is None:
        raise RuntimeError("Models are not trained. Run python backend/train_classical.py first.")
    vector = bundle["vectorizer"].transform([normalize_text(text)])
    model_names = list(bundle["models"]) if model_name == "all" else [model_name]
    results = []
    for name in model_names:
        model = bundle["models"][name]
        prediction = int(model.predict(vector)[0])
        label = "FAKE" if prediction == 0 else "REAL"  # Verify source dataset convention.
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(vector)[0]))
        results.append({
            "model": name,
            "label": label,
            "confidence": round(confidence, 4) if confidence is not None else None,
            "confidence_note": "Model score, not probability of factual truth."
        })
    fake_votes = sum(item["label"] == "FAKE" for item in results)
    real_votes = sum(item["label"] == "REAL" for item in results)
    overall = "UNCERTAIN" if fake_votes == real_votes else ("FAKE" if fake_votes > real_votes else "REAL")
    return {
        "overall": overall,
        "model_count": len(results),
        "results": results,
        "warning": "This predicts text patterns learned from dataset labels; it does not fact-check claims."
    }
