"""The measured 800×480 recipe; frames follow H3's 17n+5 layout."""
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent


class RenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    frames: int = Field(default=124, ge=22, le=345)
    seed: int = Field(default=9175, ge=0)
    vram_limit: float = Field(default=10, gt=0)

    @field_validator("frames")
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError("frames must equal 17n+5 (22, 39, 56, …, 345)")
        return value
