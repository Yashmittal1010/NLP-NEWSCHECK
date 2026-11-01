from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.main import app
from app.schemas import PredictRequest
from app.text import normalize_text

client = TestClient(app)

def test_normalization_preserves_negation():
    value = normalize_text("<p>NOT true! https://example.com</p>")
    assert "not true" in value and "https://" not in value and "url" in value

def test_request_validation():
    assert PredictRequest(text="A sufficiently long example text", model="all").model == "all"
    try:
        PredictRequest(text="A sufficiently long example text", model="bad")
    except ValidationError:
        pass
    else:
        raise AssertionError("Unsupported model should be rejected")

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert "models_trained" in response.json()

def test_short_input_rejected():
    assert client.post("/predict", json={"text": "tiny", "model": "all"}).status_code == 422
