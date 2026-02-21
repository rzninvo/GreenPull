from fastapi import APIRouter, Depends, HTTPException
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.db_models import Job, JobStatus
from app.models.schemas import (
    AnalyzeRequest,
    DetectionResult,
    EmissionsDetail,
    JobCreatedResponse,
    JobStatusResponse,
)

router = APIRouter()


def get_queue() -> Queue:
    return Queue(connection=Redis.from_url(settings.REDIS_URL))


@router.post("/analyze", response_model=JobCreatedResponse)
def analyze_repo(req: AnalyzeRequest, db: Session = Depends(get_db)):
    job = Job(repo_url=req.repo_url)
    db.add(job)
    db.commit()
    db.refresh(job)

    q = get_queue()
    q.enqueue(
        "app.worker.tasks.run_pipeline",
        job.id,
        req.repo_url,
        req.patch_type,
        req.country_iso_code,
        req.max_training_seconds,
        job_timeout=600,
    )

    return JobCreatedResponse(
        job_id=job.id,
        status=JobStatus.QUEUED,
        message="Job queued. Poll GET /api/jobs/{job_id} for status.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    detection = None
    if job.entrypoint_file:
        detection = DetectionResult(
            entrypoint_file=job.entrypoint_file,
            run_command=job.run_command,
            framework=job.framework,
            reasoning=job.detection_reasoning,
        )

    baseline = None
    if job.baseline_emissions_kg is not None:
        baseline = EmissionsDetail(
            emissions_kg=job.baseline_emissions_kg,
            energy_kwh=job.baseline_energy_kwh,
            duration_s=job.baseline_duration_s,
            cpu_energy=job.baseline_cpu_energy,
            gpu_energy=job.baseline_gpu_energy,
            water_l=job.baseline_water_l,
            cpu_model=job.baseline_cpu_model,
            gpu_model=job.baseline_gpu_model,
        )

    optimized = None
    if job.optimized_emissions_kg is not None:
        optimized = EmissionsDetail(
            emissions_kg=job.optimized_emissions_kg,
            energy_kwh=job.optimized_energy_kwh,
            duration_s=job.optimized_duration_s,
            cpu_energy=job.optimized_cpu_energy,
            gpu_energy=job.optimized_gpu_energy,
            water_l=job.optimized_water_l,
        )

    savings = None
    if job.emissions_saved_kg is not None:
        savings = {
            "emissions_saved_kg": job.emissions_saved_kg,
            "emissions_saved_pct": job.emissions_saved_pct,
            "energy_saved_kwh": job.energy_saved_kwh,
            "energy_saved_pct": job.energy_saved_pct,
        }

    return JobStatusResponse(
        job_id=job.id,
        repo_url=job.repo_url,
        status=job.status,
        created_at=job.created_at,
        detection=detection,
        baseline=baseline,
        optimized=optimized,
        patch_type=job.patch_type,
        patch_diff=job.patch_diff,
        savings=savings,
        error_message=job.error_message,
    )


@router.get("/jobs")
def list_jobs(db: Session = Depends(get_db)):
    jobs = db.query(Job).order_by(Job.created_at.desc()).limit(50).all()
    return [
        {
            "job_id": j.id,
            "repo_url": j.repo_url,
            "status": j.status,
            "created_at": j.created_at,
        }
        for j in jobs
    ]


@router.get("/health")
def health():
    return {"status": "ok"}
