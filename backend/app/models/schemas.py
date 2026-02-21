from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# --- Requests ---


class AnalyzeRequest(BaseModel):
    repo_url: str
    patch_type: str = "amp"  # "amp", "lora", or "both"
    country_iso_code: str = "USA"
    max_training_seconds: int = 300


# --- Responses ---


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    message: str


class EmissionsDetail(BaseModel):
    emissions_kg: Optional[float] = None
    energy_kwh: Optional[float] = None
    duration_s: Optional[float] = None
    cpu_energy: Optional[float] = None
    gpu_energy: Optional[float] = None
    water_l: Optional[float] = None
    cpu_model: Optional[str] = None
    gpu_model: Optional[str] = None


class DetectionResult(BaseModel):
    entrypoint_file: Optional[str] = None
    run_command: Optional[str] = None
    framework: Optional[str] = None
    reasoning: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    repo_url: str
    status: str
    created_at: datetime
    detection: Optional[DetectionResult] = None
    baseline: Optional[EmissionsDetail] = None
    optimized: Optional[EmissionsDetail] = None
    patch_type: Optional[str] = None
    patch_diff: Optional[str] = None
    savings: Optional[dict] = None
    error_message: Optional[str] = None
