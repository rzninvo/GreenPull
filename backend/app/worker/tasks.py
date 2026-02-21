import logging
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

logger = logging.getLogger("greenpull")


def _update_job(db: Session, job_id: str, **kwargs):
    db.query(Job).filter(Job.id == job_id).update(kwargs)
    db.commit()


def _install_dependencies(clone_path: Path):
    """Install repo dependencies before running training."""
    req_file = clone_path / "requirements.txt"
    if req_file.exists():
        logger.info(f"[Deps] Installing from requirements.txt")
        result = subprocess.run(
            ["pip", "install", "-r", str(req_file)],
            cwd=str(clone_path),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.warning(f"[Deps] pip install failed:\n{result.stderr[-1000:]}")
        else:
            logger.info("[Deps] pip install succeeded")
    elif (clone_path / "pyproject.toml").exists():
        logger.info(f"[Deps] Installing from pyproject.toml")
        subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=str(clone_path),
            capture_output=True,
            timeout=300,
        )
    elif (clone_path / "setup.py").exists():
        logger.info(f"[Deps] Installing from setup.py")
        subprocess.run(
            ["pip", "install", "-e", "."],
            cwd=str(clone_path),
            capture_output=True,
            timeout=300,
        )
    else:
        logger.info("[Deps] No requirements.txt, pyproject.toml, or setup.py found")


def run_pipeline(
    job_id: str,
    repo_url: str,
    patch_type: str = "amp",
    country_iso_code: str = "USA",
    max_training_seconds: int = 300,
):
    # Setup logging for this run
    _setup_logging()

    logger.info("=" * 60)
    logger.info(f"[Pipeline] START job={job_id[:8]}...")
    logger.info(f"[Pipeline] repo={repo_url}")
    logger.info(f"[Pipeline] patch={patch_type}, country={country_iso_code}, timeout={max_training_seconds}s")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        # --- 1. Clone ---
        logger.info("[Pipeline] STEP 1: Cloning repository")
        _update_job(db, job_id, status=JobStatus.CLONING)
        analyzer = RepoAnalyzer()
        clone_path = analyzer.clone_repo(repo_url, job_id)
        _update_job(db, job_id, clone_path=str(clone_path))

        # --- 1b. Install dependencies ---
        logger.info("[Pipeline] STEP 1b: Installing dependencies")
        _install_dependencies(clone_path)

        # --- 2. Detect entrypoint (regex + iterative GPT) ---
        logger.info("[Pipeline] STEP 2: Detecting training entrypoint")
        _update_job(db, job_id, status=JobStatus.ANALYZING)
        candidates = analyzer.scan_for_candidates(clone_path)
        detection = analyzer.analyze_scripts_iteratively(candidates, clone_path)

        if detection.entrypoint_file is None:
            logger.error("[Pipeline] FAILED: Could not identify training entrypoint")
            _update_job(
                db, job_id,
                status=JobStatus.FAILED,
                error_message="Could not identify a training entrypoint in this repo.",
                detection_reasoning=detection.reasoning,
            )
            return

        logger.info(f"[Pipeline] Detected entrypoint: {detection.entrypoint_file}")
        logger.info(f"[Pipeline] Run command: {detection.run_command}")
        logger.info(f"[Pipeline] Framework: {detection.framework}")

        _update_job(
            db, job_id,
            entrypoint_file=detection.entrypoint_file,
            run_command=detection.run_command,
            framework=detection.framework,
            detection_reasoning=detection.reasoning,
        )

        # --- 3. Run baseline ---
        logger.info("[Pipeline] STEP 3: Running BASELINE measurement")
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

        logger.info(f"[Pipeline] Baseline: {baseline.emissions_kg:.6f} kg CO2, "
                     f"{baseline.energy_kwh:.6f} kWh, {baseline.duration_s:.1f}s")

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
        logger.info(f"[Pipeline] STEP 4: Applying {patch_type} patch")
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
        logger.info("[Pipeline] STEP 5: Running OPTIMIZED measurement")
        _update_job(db, job_id, status=JobStatus.RUNNING_OPTIMIZED)
        results_optimized = Path(settings.RESULTS_DIR) / job_id / "optimized"

        optimized = runner.run_with_tracking(
            run_command=detection.run_command,
            working_dir=clone_path,
            results_dir=results_optimized,
            project_name=f"greenpull-optimized-{job_id[:8]}",
            timeout=max_training_seconds,
        )

        logger.info(f"[Pipeline] Optimized: {optimized.emissions_kg:.6f} kg CO2, "
                     f"{optimized.energy_kwh:.6f} kWh, {optimized.duration_s:.1f}s")

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

        logger.info("=" * 60)
        logger.info("[Pipeline] STEP 6: RESULTS")
        logger.info(f"  Emissions saved: {em_saved:.6f} kg CO2 ({em_pct:.1f}%)")
        logger.info(f"  Energy saved:    {en_saved:.6f} kWh ({en_pct:.1f}%)")
        logger.info(f"  Water saved:     {baseline.water_l - optimized.water_l:.4f} L")
        logger.info("=" * 60)

        _update_job(
            db, job_id,
            status=JobStatus.COMPLETED,
            emissions_saved_kg=em_saved,
            emissions_saved_pct=em_pct,
            energy_saved_kwh=en_saved,
            energy_saved_pct=en_pct,
        )

        logger.info(f"[Pipeline] DONE job={job_id[:8]}")

    except Exception as e:
        logger.error(f"[Pipeline] FAILED: {type(e).__name__}: {e}")
        logger.error(traceback.format_exc())
        _update_job(
            db, job_id,
            status=JobStatus.FAILED,
            error_message=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[-1000:]}",
        )
    finally:
        db.close()


def _setup_logging():
    """Configure the greenpull logger for debug output."""
    log = logging.getLogger("greenpull")
    if not log.handlers:
        log.setLevel(logging.DEBUG if settings.DEBUG else logging.INFO)
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        log.addHandler(handler)
