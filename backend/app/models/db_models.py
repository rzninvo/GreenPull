import enum
import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, Text

from app.core.database import Base


class JobStatus(str, enum.Enum):
    QUEUED = "queued"
    CLONING = "cloning"
    ANALYZING = "analyzing"
    RUNNING_BASELINE = "running_baseline"
    PATCHING = "patching"
    RUNNING_OPTIMIZED = "running_optimized"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    repo_url = Column(String, nullable=False)
    status = Column(String, default=JobStatus.QUEUED)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Detection results
    entrypoint_file = Column(String, nullable=True)
    run_command = Column(String, nullable=True)
    framework = Column(String, nullable=True)
    detection_reasoning = Column(Text, nullable=True)

    # Baseline results
    baseline_emissions_kg = Column(Float, nullable=True)
    baseline_energy_kwh = Column(Float, nullable=True)
    baseline_duration_s = Column(Float, nullable=True)
    baseline_cpu_energy = Column(Float, nullable=True)
    baseline_gpu_energy = Column(Float, nullable=True)
    baseline_water_l = Column(Float, nullable=True)
    baseline_cpu_model = Column(String, nullable=True)
    baseline_gpu_model = Column(String, nullable=True)

    # Patch info
    patch_type = Column(String, nullable=True)
    patch_diff = Column(Text, nullable=True)

    # Optimized results
    optimized_emissions_kg = Column(Float, nullable=True)
    optimized_energy_kwh = Column(Float, nullable=True)
    optimized_duration_s = Column(Float, nullable=True)
    optimized_cpu_energy = Column(Float, nullable=True)
    optimized_gpu_energy = Column(Float, nullable=True)
    optimized_water_l = Column(Float, nullable=True)

    # Savings
    emissions_saved_kg = Column(Float, nullable=True)
    emissions_saved_pct = Column(Float, nullable=True)
    energy_saved_kwh = Column(Float, nullable=True)
    energy_saved_pct = Column(Float, nullable=True)

    # Error / metadata
    error_message = Column(Text, nullable=True)
    clone_path = Column(String, nullable=True)
