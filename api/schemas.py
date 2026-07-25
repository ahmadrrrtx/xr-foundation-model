"""Pydantic schemas for XRFM API (v1.0.0)."""

from pydantic import BaseModel, Field


class CompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32768)
    max_new_tokens: int = Field(default=50, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    top_k: int | None = Field(default=None, ge=1, le=1000)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    stop: list[str] | None = Field(default=None)
    stream: bool = Field(default=False)


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str = "length"


class CompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str = "xrfm-completion"
    object: str = "text_completion"
    created: int = 0
    model: str = "xrfm"
    choices: list[CompletionChoice]
    usage: CompletionUsage


class TokenizeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=32768)


class TokenizeResponse(BaseModel):
    tokens: list[int]
    count: int


class DetokenizeRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=32768)


class DetokenizeResponse(BaseModel):
    text: str


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "xrfm"
    vocab_size: int
    max_seq_len: int
    parameter_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    gpu_available: bool
