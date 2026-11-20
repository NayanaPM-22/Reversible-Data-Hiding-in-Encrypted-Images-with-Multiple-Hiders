from pydantic import BaseModel, field_validator
from typing import List

MIN_HIDERS = 2
MAX_HIDERS = 8
MAX_IMAGE_DIM = 1024
MIN_IMAGE_DIM = 64
MAX_SECRET_CHARS = 2000


class SetupRequest(BaseModel):
    num_hiders: int

    @field_validator('num_hiders')
    @classmethod
    def validate_num_hiders(cls, v):
        if v < MIN_HIDERS:
            raise ValueError(
                f"Number of hiders must be at least {MIN_HIDERS}. "
                f"n={v} provides no meaningful secret sharing security."
            )
        if v > MAX_HIDERS:
            raise ValueError(f"Number of hiders cannot exceed {MAX_HIDERS} in this demo.")
        return v


class HiderSecret(BaseModel):
    hider_id: int
    secret_text: str

    @field_validator('secret_text')
    @classmethod
    def validate_secret_length(cls, v):
        if not v or not v.strip():
            raise ValueError("Secret message cannot be empty.")
        if len(v) > MAX_SECRET_CHARS:
            raise ValueError(f"Secret too long ({len(v)} chars). Max {MAX_SECRET_CHARS}.")
        return v


class HidersEmbedRequest(BaseModel):
    session_id: str
    secrets: List[HiderSecret]