import subprocess
import traceback
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import Job, JobStatus
from app.services.carbon_runner import CarbonRunner
from app.services.patch_engine import PatchEngine
from app.services.repo_analyzer import RepoAnalyzer


def _update_job(db: Session, job_id: str, **kwargs):
    db.query(Job).filter(Job.id == job_id).update(kwargs)
    db.commit()


def _install_dependencies(clone_path: Path):
    """Install repo dependencies before running training."""
    req_file = clone_path / "requirements.txt"
    if req_file.exists():
        subprocess.run(
            ["pip", "install", "-r", str(req_file)],
            cwd=str(clone_path),
            capture_output=True,
            timeout=300,
        )
    # Also check for pyproject.toml with pip install .
    elif (clone_path / "pyproject.toml").exists():
        subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=str(clone_path),
            capture_output=True,
            timeout=300,
        )
    elif (clone_path / "setup.py").exists():
        subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=str(clone_path),
            capture_output=True,
            timeout=300,
        )


def run_pipeline(
    job_id: str,
    repo_url: str,
    patch_type: str = "amp",
    country_iso_code: str = "USA",
    max_training_seconds: int = 300,
):
    db = SessionLocal()
    try:
        # --- 1. Clone ---
        _update_job(db, job_id, status=JobStatus.CLONING)
        analyzer = RepoAnalyzer()
        clone_path = analyzer.clone_repo(repo_url, job_id)
        _update_job(db, job_id, clone_path=str(clone_path))

        # --- 1b. Install dependencies ---
        _install_dependencies(clone_path)

        # --- 2. Detect entrypoint (regex + iterative GPT) ---
        _update_job(db, job_id, status=JobStatus.ANALYZING)
        candidates = analyzer.scan_for_candidates(clone_path)
        detection = analyzer.analyze_scripts_iteratively(candidates, clone_path)

        if detection.entrypoint_file is None:
            _update_job(
                db, job_id,
                status=JobStatus.FAILED,
                error_message="Could not identify a training entrypoint in this repo.",
                detection_reasoning=detection.reasoning,
            )
            return

        _update_job(
            db, job_id,
            entrypoint_file=detection.entrypoint_file,
            run_command=detection.run_command,
            framework=detection.framework,
            detection_reasoning=detection.reasoning,
        )

        # --- 3. Run baseline ---
        _update_job(db, job_id, status=JobStatus.RUNNING_BASELINE)
        runner = CarbonRunner(country_iso_code=country_iso_code)
        results_baseline = Path(settings.RESULTS_DIR) / job_id / "baseline"

        baseline = runner.run_with_tracking(
            run_command=detection.run_command,
            working_dir=clone_path,
            results_dir=results_baseline,
            project_name=f"greenpull-baseline-{job_id[:8]}",
            timeout=max_training_seconds,
        )

        _update_job(
            db, job_id,
            baseline_emissions_kg=baseline.emissions_kg,
            baseline_energy_kwh=baseline.energy_kwh,
            baseline_duration_s=baseline.duration_s,
            baseline_cpu_energy=baseline.cpu_energy,
            baseline_gpu_energy=baseline.gpu_energy,
            baseline_water_l=baseline.water_l,
            baseline_cpu_model=baseline.cpu_model,
            baseline_gpu_model=baseline.gpu_model,
        )

        # --- 4. Apply patch ---
        _update_job(db, job_id, status=JobStatus.PATCHING)
        patcher = PatchEngine()
        _, diff = patcher.apply_patch(
            repo_path=clone_path,
            entrypoint_file=detection.entrypoint_file,
            framework=detection.framework or "unknown",
            patch_type=patch_type,
        )
        _update_job(db, job_id, patch_type=patch_type, patch_diff=diff)

        # --- 5. Run optimized ---
        _update_job(db, job_id, status=JobStatus.RUNNING_OPTIMIZED)
        results_optimized = Path(settings.RESULTS_DIR) / job_id / "optimized"

        optimized = runner.run_with_tracking(
            run_command=detection.run_command,
            working_dir=clone_path,
            results_dir=results_optimized,
            project_name=f"greenpull-optimized-{job_id[:8]}",
            timeout=max_training_seconds,
        )

        _update_job(
            db, job_id,
            optimized_emissions_kg=optimized.emissions_kg,
            optimized_energy_kwh=optimized.energy_kwh,
            optimized_duration_s=optimized.duration_s,
            optimized_cpu_energy=optimized.cpu_energy,
            optimized_gpu_energy=optimized.gpu_energy,
            optimized_water_l=optimized.water_l,
        )

        # --- 6. Compute savings ---
        em_saved = baseline.emissions_kg - optimized.emissions_kg
        em_pct = (em_saved / baseline.emissions_kg * 100) if baseline.emissions_kg > 0 else 0
        en_saved = baseline.energy_kwh - optimized.energy_kwh
        en_pct = (en_saved / baseline.energy_kwh * 100) if baseline.energy_kwh > 0 else 0

        _update_job(
            db, job_id,
            status=JobStatus.COMPLETED,
            emissions_saved_kg=em_saved,
            emissions_saved_pct=em_pct,
            energy_saved_kwh=en_saved,
            energy_saved_pct=en_pct,
        )

    except Exception as e:
        _update_job(
            db, job_id,
            status=JobStatus.FAILED,
            error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-1000:]}",
        )
    finally:
        db.close()
