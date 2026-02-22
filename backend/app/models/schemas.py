from datetime import datetime
from typing import Optional

from pydantic import BaseModel


# --- Requests ---


class AnalyzeRequest(BaseModel):
    repo_url: str
    patch_type: str = "amp"  # "amp", "lora", or "both"
    country_iso_code: str = "DEU"  # default Germany (EU hackathon)


# --- Responses ---


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str
    message: str


class EmissionsDetail(BaseModel):
    emissions_kg: Optional[float] = None
    energy_kwh: Optional[float] = None
    duration_s: Optional[float] = None
    power_w: Optional[float] = None
    cpu_energy_kwh: Optional[float] = None
    gpu_energy_kwh: Optional[float] = None
    memory_energy_kwh: Optional[float] = None
    cpu_model: Optional[str] = None
    gpu_model: Optional[str] = None


class DetectionResult(BaseModel):
    entrypoint_file: Optional[str] = None
    run_command: Optional[str] = None
    framework: Optional[str] = None
    reasoning: Optional[str] = None


class TrainingConfigResponse(BaseModel):
    model_type: Optional[str] = None
    model_name: Optional[str] = None
    parameter_count_millions: Optional[float] = None
    framework: Optional[str] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    dataset_size_estimate: Optional[str] = None
    estimated_runtime_hours: Optional[float] = None
    gpu_type: Optional[str] = None
    num_gpus: Optional[int] = None
    reasoning: Optional[str] = None


class ComparisonMetricsResponse(BaseModel):
    tree_months: Optional[float] = None
    car_km: Optional[float] = None
    smartphone_charges: Optional[float] = None
    streaming_hours: Optional[float] = None
    flight_fraction: Optional[float] = None
    led_bulb_hours: Optional[float] = None


class CarbonIntensityInfo(BaseModel):
    value: Optional[float] = None
    zone: Optional[str] = None
    source: Optional[str] = None  # "api" or "static_fallback"
    datetime_utc: Optional[str] = None


class GreenWindowResponse(BaseModel):
    best_window_start: Optional[str] = None
    best_window_end: Optional[str] = None
    best_intensity: Optional[float] = None
    current_intensity: Optional[float] = None
    savings_pct: Optional[float] = None


class RegionRecommendationResponse(BaseModel):
    current_zone: Optional[str] = None
    current_intensity: Optional[float] = None
    recommended_provider: Optional[str] = None
    recommended_region_code: Optional[str] = None
    recommended_region_name: Optional[str] = None
    recommended_country: Optional[str] = None
    recommended_city: Optional[str] = None
    recommended_intensity: Optional[float] = None
    savings_pct: Optional[float] = None


class WaterUsageResponse(BaseModel):
    baseline_liters: Optional[float] = None
    optimized_liters: Optional[float] = None
    wue: Optional[float] = None


class PatchedFileInfo(BaseModel):
    file_path: str
    role: str
    optimization: str


class CreatePRRequest(BaseModel):
    github_token: str
    title: str = ""
    body: str = ""
    branch_name: str = "greenpull/optimization"
    base_branch: str = "main"


class CreatePRResponse(BaseModel):
    pr_number: int
    pr_url: str


class JobStatusResponse(BaseModel):
    job_id: str
    repo_url: str
    status: str
    created_at: datetime
    estimation_method: Optional[str] = None
    detection: Optional[DetectionResult] = None
    training_config: Optional[TrainingConfigResponse] = None
    baseline: Optional[EmissionsDetail] = None
    optimized: Optional[EmissionsDetail] = None
    patch_type: Optional[str] = None
    patch_diff: Optional[str] = None
    savings: Optional[dict] = None
    comparisons: Optional[ComparisonMetricsResponse] = None
    carbon_intensity_info: Optional[CarbonIntensityInfo] = None
    green_window: Optional[GreenWindowResponse] = None
    region_recommendation: Optional[RegionRecommendationResponse] = None
    water_usage: Optional[WaterUsageResponse] = None
    patched_files: Optional[list[PatchedFileInfo]] = None
    error_message: Optional[str] = None
