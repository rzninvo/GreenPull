import json

from fastapi import APIRouter, Depends, HTTPException
from redis import Redis
from rq import Queue
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db
from app.models.db_models import Job, JobStatus
from app.models.schemas import (
    AnalyzeRequest,
    ComparisonMetricsResponse,
    CreatePRRequest,
    CreatePRResponse,
    DetectionResult,
    EmissionsDetail,
    JobCreatedResponse,
    JobStatusResponse,
    TrainingConfigResponse,
)
from app.services.github_service import GitHubService

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

    training_config = None
    if job.training_config_json:
        tc = json.loads(job.training_config_json)
        training_config = TrainingConfigResponse(
            model_type=tc.get("model_type"),
            model_name=tc.get("model_name"),
            parameter_count_millions=tc.get("parameter_count_millions"),
            framework=job.framework,
            epochs=tc.get("epochs"),
            batch_size=tc.get("batch_size"),
            dataset_size_estimate=tc.get("dataset_size_estimate"),
            estimated_runtime_hours=tc.get("estimated_runtime_hours"),
            gpu_type=tc.get("gpu_type"),
            num_gpus=tc.get("num_gpus"),
            reasoning=tc.get("reasoning"),
        )

    baseline = None
    if job.baseline_emissions_kg is not None:
        baseline = EmissionsDetail(
            emissions_kg=job.baseline_emissions_kg,
            energy_kwh=job.baseline_energy_kwh,
            duration_s=job.baseline_duration_s,
            cpu_energy_kwh=job.baseline_cpu_energy,
            gpu_energy_kwh=job.baseline_gpu_energy,
            cpu_model=job.baseline_cpu_model,
            gpu_model=job.baseline_gpu_model,
        )

    optimized = None
    if job.optimized_emissions_kg is not None:
        optimized = EmissionsDetail(
            emissions_kg=job.optimized_emissions_kg,
            energy_kwh=job.optimized_energy_kwh,
            duration_s=job.optimized_duration_s,
            cpu_energy_kwh=job.optimized_cpu_energy,
            gpu_energy_kwh=job.optimized_gpu_energy,
        )

    savings = None
    if job.emissions_saved_kg is not None:
        savings = {
            "emissions_saved_kg": job.emissions_saved_kg,
            "emissions_saved_pct": job.emissions_saved_pct,
            "energy_saved_kwh": job.energy_saved_kwh,
            "energy_saved_pct": job.energy_saved_pct,
        }

    comparisons = None
    if job.tree_months is not None:
        comparisons = ComparisonMetricsResponse(
            tree_months=job.tree_months,
            car_km=job.car_km,
            smartphone_charges=job.smartphone_charges,
            streaming_hours=job.streaming_hours,
            flight_fraction=job.flight_fraction,
            led_bulb_hours=job.led_bulb_hours,
        )

    return JobStatusResponse(
        job_id=job.id,
        repo_url=job.repo_url,
        status=job.status,
        created_at=job.created_at,
        estimation_method=job.estimation_method,
        detection=detection,
        training_config=training_config,
        baseline=baseline,
        optimized=optimized,
        patch_type=job.patch_type,
        patch_diff=job.patch_diff,
        savings=savings,
        comparisons=comparisons,
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


@router.post("/jobs/{job_id}/create-pr", response_model=CreatePRResponse)
def create_pull_request(job_id: str, req: CreatePRRequest, db: Session = Depends(get_db)):
    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job is not completed yet")

    if not job.patched_code or not job.entrypoint_file:
        raise HTTPException(status_code=400, detail="No patched code available for this job")

    # Build default title/body if not provided
    patch_label = {
        "amp": "AMP Mixed Precision",
        "lora": "LoRA Fine-Tuning",
        "both": "AMP + LoRA",
    }.get(job.patch_type or "amp", job.patch_type or "Optimization")

    title = req.title or f"GreenPull: Add {patch_label} to {job.entrypoint_file}"

    if not req.body:
        body_parts = [f"## GreenPull Optimization\n"]
        body_parts.append(f"**File:** `{job.entrypoint_file}`")
        body_parts.append(f"**Technique:** {patch_label}\n")
        if job.emissions_saved_kg is not None:
            body_parts.append(f"### Estimated Savings")
            body_parts.append(f"- **CO2 reduced:** {job.emissions_saved_kg:.4f} kg ({job.emissions_saved_pct:.1f}%)")
            body_parts.append(f"- **Energy saved:** {job.energy_saved_kwh:.4f} kWh ({job.energy_saved_pct:.1f}%)")
        body_parts.append(f"\n---\n*Generated by [GreenPull](https://github.com) — carbon-aware ML optimization*")
        body = "\n".join(body_parts)
    else:
        body = req.body

    try:
        gh = GitHubService(token=req.github_token, repo_url=job.repo_url)
        result = gh.create_optimization_pr(
            file_path=job.entrypoint_file,
            patched_code=job.patched_code,
            title=title,
            body=body,
            branch_name=req.branch_name,
            base_branch=req.base_branch,
        )
        return CreatePRResponse(pr_number=result["number"], pr_url=result["url"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def health():
    return {"status": "ok"}
