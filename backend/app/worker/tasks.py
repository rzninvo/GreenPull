import json
import logging
import shutil
import traceback
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.db_models import Job, JobStatus
from app.services.electricity_maps_service import ElectricityMapsService
from app.services.green_estimator import GreenEstimator, TrainingConfig
from app.services.patch_engine import PatchEngine
from app.services.repo_analyzer import RepoAnalyzer

logger = logging.getLogger("greenpull")


def _update_job(db: Session, job_id: str, **kwargs):
    db.query(Job).filter(Job.id == job_id).update(kwargs)
    db.commit()


def _dict_to_training_config(config: dict, framework: str) -> TrainingConfig:
    """Convert GPT's JSON dict to a TrainingConfig dataclass."""
    return TrainingConfig(
        model_type=config.get("model_type", "unknown"),
        model_name=config.get("model_name"),
        parameter_count=config.get("parameter_count_millions", 100.0),
        framework=framework,
        epochs=config.get("epochs", 3),
        batch_size=config.get("batch_size", 32),
        dataset_size_estimate=config.get("dataset_size_estimate", "medium (10K-1M)"),
        estimated_runtime_hours=config.get("estimated_runtime_hours", 1.0),
        gpu_required=config.get("gpu_required", True),
        gpu_type=config.get("gpu_type", "V100"),
        num_gpus=config.get("num_gpus", 1),
        cpu_type=config.get("cpu_type", "server_generic"),
        num_cpu_cores=config.get("num_cpu_cores", 8),
        memory_gb=config.get("memory_gb", 16.0),
        reasoning=config.get("reasoning", ""),
    )


def run_pipeline(
    job_id: str,
    repo_url: str,
    patch_type: str = "amp",
    country_iso_code: str = "DEU",
):
    _setup_logging()

    logger.info("=" * 60)
    logger.info(f"[Pipeline] START job={job_id[:8]}...")
    logger.info(f"[Pipeline] repo={repo_url}")
    logger.info(f"[Pipeline] patch={patch_type}, country={country_iso_code}")
    logger.info("=" * 60)

    db = SessionLocal()
    try:
        # --- 1. Clone ---
        logger.info("[Pipeline] STEP 1: Cloning repository")
        _update_job(db, job_id, status=JobStatus.CLONING)
        analyzer = RepoAnalyzer()
        clone_path = analyzer.clone_repo(repo_url, job_id)
        _update_job(db, job_id, clone_path=str(clone_path))

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

        # --- 2b. Identify patchable files ---
        logger.info("[Pipeline] STEP 2b: Identifying patchable files")
        patchable_files = analyzer.identify_patchable_files(
            repo_path=clone_path,
            entrypoint_file=detection.entrypoint_file,
            framework=detection.framework or "unknown",
            patch_type=patch_type,
        )

        # --- 3. Extract training config via GPT ---
        logger.info("[Pipeline] STEP 3: Extracting training configuration")
        _update_job(db, job_id, status=JobStatus.EXTRACTING_CONFIG)

        entrypoint_path = clone_path / detection.entrypoint_file
        original_code = entrypoint_path.read_text(errors="ignore")
        dep_context = analyzer._get_dependency_context(clone_path)
        config_context = analyzer._gather_config_files(clone_path)
        imports_context = analyzer._resolve_local_imports(clone_path, detection.entrypoint_file)
        readme_context = analyzer._get_readme_context(clone_path)

        if config_context:
            logger.info(f"[Pipeline] Found config files ({len(config_context)} chars)")
        if imports_context:
            logger.info(f"[Pipeline] Resolved local imports ({len(imports_context)} chars)")
        if readme_context:
            logger.info(f"[Pipeline] Found README ({len(readme_context)} chars)")

        baseline_config = analyzer.extract_training_config(
            code=original_code,
            entrypoint_file=detection.entrypoint_file,
            framework=detection.framework or "unknown",
            dep_context=dep_context,
            config_context=config_context,
            imports_context=imports_context,
            readme_context=readme_context,
        )

        logger.info(f"[Pipeline] Extracted config:\n{json.dumps(baseline_config, indent=2)}")
        _update_job(db, job_id, training_config_json=json.dumps(baseline_config))

        # --- 3b. Fetch real-time carbon intensity from Electricity Maps ---
        logger.info("[Pipeline] STEP 3b: Fetching real-time carbon intensity")
        emaps = ElectricityMapsService()
        ci_result = emaps.get_carbon_intensity(country_iso_code)
        logger.info(
            f"[Pipeline] Carbon intensity: {ci_result.carbon_intensity} gCO2/kWh "
            f"(zone={ci_result.zone}, source={ci_result.source})"
        )
        _update_job(
            db, job_id,
            carbon_intensity_gco2_kwh=ci_result.carbon_intensity,
            carbon_intensity_source=ci_result.source,
            carbon_intensity_zone=ci_result.zone,
            carbon_intensity_datetime=ci_result.datetime_utc,
        )

        # --- 4. Estimate baseline emissions ---
        logger.info("[Pipeline] STEP 4: Estimating BASELINE emissions")
        _update_job(db, job_id, status=JobStatus.ESTIMATING_BASELINE)

        estimator = GreenEstimator(
            country_code=country_iso_code,
            carbon_intensity_override=ci_result.carbon_intensity,
        )
        training_cfg = _dict_to_training_config(baseline_config, detection.framework or "unknown")
        hw_config = estimator.resolve_hardware(training_cfg)
        baseline_result = estimator.estimate_emissions(training_cfg, hw_config)

        logger.info(f"[Pipeline] Baseline: {baseline_result.emissions_kg:.6f} kg CO2, "
                     f"{baseline_result.energy_kwh:.6f} kWh, {baseline_result.duration_s:.1f}s")

        _update_job(
            db, job_id,
            baseline_emissions_kg=baseline_result.emissions_kg,
            baseline_energy_kwh=baseline_result.energy_kwh,
            baseline_duration_s=baseline_result.duration_s,
            baseline_cpu_energy=baseline_result.cpu_energy_kwh,
            baseline_gpu_energy=baseline_result.gpu_energy_kwh,
            baseline_cpu_model=hw_config.cpu_name,
            baseline_gpu_model=hw_config.gpu_name,
            hardware_assumptions=json.dumps({
                "gpu_tdp_w": hw_config.gpu_tdp_w,
                "gpu_name": hw_config.gpu_name,
                "cpu_tdp_w": hw_config.cpu_tdp_w,
                "cpu_name": hw_config.cpu_name,
                "num_gpus": hw_config.num_gpus,
                "num_cpu_cores": hw_config.num_cpu_cores,
                "memory_gb": hw_config.memory_gb,
                "pue": hw_config.pue,
            }),
            country_code=country_iso_code,
        )

        # --- 5. Apply patch (multi-file) ---
        logger.info(f"[Pipeline] STEP 5: Applying {patch_type} patch to {len(patchable_files)} file(s)")
        _update_job(db, job_id, status=JobStatus.PATCHING)
        patcher = PatchEngine()

        if len(patchable_files) > 1:
            multi_result = patcher.apply_multi_file_patch(
                patchable_files=patchable_files,
                framework=detection.framework or "unknown",
                patch_type=patch_type,
                dep_context=dep_context,
                config_context=config_context,
                readme_context=readme_context,
            )
            diff = multi_result.combined_diff
            # Entrypoint patched code for backward compat
            ep_patch = next(
                (fp for fp in multi_result.file_patches if fp.role == "entrypoint"), None
            )
            patched_code = ep_patch.patched_code if ep_patch else original_code
            patched_files_data = json.dumps([
                {
                    "file_path": fp.file_path,
                    "role": fp.role,
                    "optimization": fp.optimization,
                    "patched_code": fp.patched_code,
                }
                for fp in multi_result.file_patches
            ])
        else:
            patched_code, diff = patcher.apply_patch(
                repo_path=clone_path,
                entrypoint_file=detection.entrypoint_file,
                framework=detection.framework or "unknown",
                patch_type=patch_type,
                dep_context=dep_context,
                config_context=config_context,
                imports_context=imports_context,
                readme_context=readme_context,
            )
            # Rewrite diff with git-style headers for consistent frontend parsing
            if patched_code and patched_code != original_code:
                diff = PatchEngine._make_file_diff(detection.entrypoint_file, original_code, patched_code)
            patched_files_data = json.dumps([{
                "file_path": detection.entrypoint_file,
                "role": "entrypoint",
                "optimization": patch_type,
                "patched_code": patched_code,
            }])

        _update_job(
            db, job_id,
            patch_type=patch_type,
            patch_diff=diff,
            patched_code=patched_code,
            patched_files_json=patched_files_data,
        )

        # --- 6. Estimate optimized emissions ---
        logger.info("[Pipeline] STEP 6: Estimating OPTIMIZED emissions")
        _update_job(db, job_id, status=JobStatus.ESTIMATING_OPTIMIZED)

        optimized_adjustments = analyzer.estimate_optimized_config(
            baseline_config=baseline_config,
            patch_type=patch_type,
            patch_diff=diff,
            original_code=original_code,
            patched_code=patched_code,
        )

        logger.info(f"[Pipeline] Optimization adjustments:\n{json.dumps(optimized_adjustments, indent=2)}")

        # Merge baseline config with adjustments
        optimized_config = baseline_config.copy()
        optimized_config["estimated_runtime_hours"] = optimized_adjustments.get(
            "estimated_runtime_hours",
            baseline_config.get("estimated_runtime_hours", 1.0) * 0.8,
        )
        if "memory_gb" in optimized_adjustments:
            optimized_config["memory_gb"] = optimized_adjustments["memory_gb"]
        if "batch_size" in optimized_adjustments:
            optimized_config["batch_size"] = optimized_adjustments["batch_size"]
        if "parameter_count_millions" in optimized_adjustments:
            optimized_config["parameter_count_millions"] = optimized_adjustments["parameter_count_millions"]

        optimized_training_cfg = _dict_to_training_config(optimized_config, detection.framework or "unknown")
        # Re-resolve hardware so memory changes from optimization are reflected
        optimized_hw = estimator.resolve_hardware(optimized_training_cfg)
        optimized_result = estimator.estimate_emissions(optimized_training_cfg, optimized_hw)

        logger.info(f"[Pipeline] Optimized: {optimized_result.emissions_kg:.6f} kg CO2, "
                     f"{optimized_result.energy_kwh:.6f} kWh, {optimized_result.duration_s:.1f}s")

        _update_job(
            db, job_id,
            optimized_emissions_kg=optimized_result.emissions_kg,
            optimized_energy_kwh=optimized_result.energy_kwh,
            optimized_duration_s=optimized_result.duration_s,
            optimized_cpu_energy=optimized_result.cpu_energy_kwh,
            optimized_gpu_energy=optimized_result.gpu_energy_kwh,
        )

        # --- 7. Compute savings + comparisons ---
        em_saved = baseline_result.emissions_kg - optimized_result.emissions_kg
        em_pct = (em_saved / baseline_result.emissions_kg * 100) if baseline_result.emissions_kg > 0 else 0
        en_saved = baseline_result.energy_kwh - optimized_result.energy_kwh
        en_pct = (en_saved / baseline_result.energy_kwh * 100) if baseline_result.energy_kwh > 0 else 0

        comparisons = estimator.compute_comparisons(em_saved, en_saved)

        logger.info("=" * 60)
        logger.info("[Pipeline] STEP 7: RESULTS")
        logger.info(f"  Emissions saved: {em_saved:.6f} kg CO2 ({em_pct:.1f}%)")
        logger.info(f"  Energy saved:    {en_saved:.6f} kWh ({en_pct:.1f}%)")
        logger.info(f"  = {comparisons.tree_months:.2f} tree-months of CO2 absorption")
        logger.info(f"  = {comparisons.car_km:.2f} km of driving")
        logger.info(f"  = {comparisons.smartphone_charges:.0f} smartphone charges")
        logger.info(f"  = {comparisons.streaming_hours:.1f} hours of video streaming")
        logger.info("=" * 60)

        _update_job(
            db, job_id,
            emissions_saved_kg=em_saved,
            emissions_saved_pct=em_pct,
            energy_saved_kwh=en_saved,
            energy_saved_pct=en_pct,
            estimation_method="green_algorithms_v1",
            tree_months=comparisons.tree_months,
            car_km=comparisons.car_km,
            smartphone_charges=comparisons.smartphone_charges,
            streaming_hours=comparisons.streaming_hours,
            flight_fraction=comparisons.flight_fraction,
            led_bulb_hours=comparisons.led_bulb_hours,
        )

        # --- 8. Green window forecast ---
        logger.info("[Pipeline] STEP 8: Fetching green window forecast")
        runtime_hours = training_cfg.estimated_runtime_hours
        green_window = emaps.get_green_window(country_iso_code, runtime_hours)
        if green_window:
            logger.info(
                f"[Pipeline] Green window: {green_window.best_window_start} to "
                f"{green_window.best_window_end} ({green_window.best_intensity} gCO2/kWh)"
            )
            _update_job(
                db, job_id,
                green_window_start=green_window.best_window_start,
                green_window_end=green_window.best_window_end,
                green_window_intensity=green_window.best_intensity,
                current_grid_intensity=green_window.current_intensity,
                green_window_savings_pct=green_window.additional_savings_pct,
            )
        else:
            logger.info("[Pipeline] Green window: forecast not available")

        # --- 9. Region recommendation ---
        logger.info("[Pipeline] STEP 9: Computing region recommendation")
        region_rec = emaps.get_region_recommendation(country_iso_code, current_ci=ci_result)
        if region_rec:
            rec = region_rec.recommended
            logger.info(
                f"[Pipeline] Recommended region: {rec.provider} {rec.region_code} "
                f"({rec.city}, {rec.country_name}) = {rec.carbon_intensity} gCO2/kWh "
                f"(saves {region_rec.savings_pct}%)"
            )
            _update_job(
                db, job_id,
                recommended_region_provider=rec.provider,
                recommended_region_code=rec.region_code,
                recommended_region_name=f"{rec.city}, {rec.country_name}" if rec.city else rec.country_name,
                recommended_region_country=rec.country_name,
                recommended_region_city=rec.city,
                recommended_region_intensity=rec.carbon_intensity,
                region_savings_pct=region_rec.savings_pct,
            )

        # --- 10. Water usage ---
        logger.info("[Pipeline] STEP 10: Estimating water usage")
        water = ElectricityMapsService.estimate_water_usage(
            baseline_energy_kwh=baseline_result.energy_kwh,
            optimized_energy_kwh=optimized_result.energy_kwh,
        )
        logger.info(
            f"[Pipeline] Water: {water.water_liters_baseline:.4f}L baseline, "
            f"{water.water_liters_optimized:.4f}L optimized (WUE={water.wue})"
        )
        _update_job(
            db, job_id,
            status=JobStatus.COMPLETED,
            water_liters_baseline=water.water_liters_baseline,
            water_liters_optimized=water.water_liters_optimized,
            water_wue=water.wue,
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
        # Clean up cloned repo from disk
        clone_dir = settings.CLONE_DIR / job_id
        if clone_dir.exists():
            shutil.rmtree(clone_dir, ignore_errors=True)
            logger.info(f"[Pipeline] Cleaned up {clone_dir}")


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
