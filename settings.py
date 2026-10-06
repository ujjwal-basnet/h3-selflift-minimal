"""Quality baseline and experimental faster, lower-memory profiles."""
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parent
PROFILES = {
    "quality": dict(steps=8, transition_step=6, width=640, height=384,
                    target_width=800, target_height=480, vram_limit=10, gpu_cap_gib=None),
    "fast": dict(steps=4, transition_step=3, width=640, height=384,
                 target_width=800, target_height=480, vram_limit=8, gpu_cap_gib=10.5),
    "lowmem": dict(steps=4, transition_step=3, width=512, height=320,
                   target_width=640, target_height=384, vram_limit=7.5, gpu_cap_gib=10.5),
}


class RenderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile: Literal["quality", "fast", "lowmem"] = "quality"
    frames: int = Field(default=124, ge=22, le=345)
    seed: int = Field(default=9175, ge=0)
    vram_limit: float | None = Field(default=None, gt=0)

    @property
    def recipe(self):
        recipe = PROFILES[self.profile].copy()
        if self.vram_limit is not None:
            recipe["vram_limit"] = self.vram_limit
        return recipe

    @field_validator("frames")
    @classmethod
    def valid_frames(cls, value):
        if value % 17 != 5:
            raise ValueError("frames must equal 17n+5 (22, 39, 56, …, 345)")
        return value
