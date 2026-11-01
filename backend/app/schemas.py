from pydantic import BaseModel, Field, field_validator

ALLOWED_MODELS = {"logistic_regression", "passive_aggressive", "naive_bayes", "all"}

class PredictRequest(BaseModel):
    text: str = Field(min_length=10, max_length=20000)
    model: str = "logistic_regression"

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Text cannot be blank.")
        return value

    @field_validator("model")
    @classmethod
    def model_is_supported(cls, value: str) -> str:
        if value not in ALLOWED_MODELS:
            raise ValueError(f"Model must be one of: {', '.join(sorted(ALLOWED_MODELS))}")
        return value
