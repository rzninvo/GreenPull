import logging
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

from codecarbon import OfflineEmissionsTracker

logger = logging.getLogger("greenpull")


@dataclass
class EmissionsResult:
    emissions_kg: float
    energy_kwh: float
    duration_s: float
    cpu_energy: float
    gpu_energy: float
    water_l: float
    cpu_model: str
    gpu_model: str


class CarbonRunner:
    def __init__(self, country_iso_code: str = "USA"):
        self.country_iso_code = country_iso_code

    def run_with_tracking(
        self,
        run_command: str,
        working_dir: Path,
        results_dir: Path,
        project_name: str = "greenpull",
        timeout: int = 300,
    ) -> EmissionsResult:
        os.makedirs(results_dir, exist_ok=True)

        logger.info(f"[CarbonRunner] command: {run_command}")
        logger.info(f"[CarbonRunner] working_dir: {working_dir}")
        logger.info(f"[CarbonRunner] country: {self.country_iso_code}, timeout: {timeout}s")

        tracker = OfflineEmissionsTracker(
            country_iso_code=self.country_iso_code,
            project_name=project_name,
            save_to_file=True,
            output_dir=str(results_dir),
            log_level="warning",
        )

        tracker.start()
        proc = None
        try:
            proc = subprocess.Popen(
                run_command,
                shell=True,
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env={**os.environ, "PYTHONPATH": str(working_dir)},
                preexec_fn=os.setsid,
            )
            stdout, stderr = proc.communicate(timeout=timeout)

            logger.debug(f"[CarbonRunner] STDOUT:\n{stdout[-3000:]}")
            if stderr:
                logger.debug(f"[CarbonRunner] STDERR:\n{stderr[-3000:]}")

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Training command failed (exit {proc.returncode}):\n"
                    f"STDOUT: {stdout[-2000:]}\n"
                    f"STDERR: {stderr[-2000:]}"
                )
        except subprocess.TimeoutExpired:
            logger.warning(f"[CarbonRunner] Timeout after {timeout}s, killing process")
            if proc:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()
        finally:
            tracker.stop()

        data = tracker.final_emissions_data

        result = EmissionsResult(
            emissions_kg=data.emissions,
            energy_kwh=data.energy_consumed,
            duration_s=data.duration,
            cpu_energy=data.cpu_energy,
            gpu_energy=data.gpu_energy,
            water_l=data.water_consumed,
            cpu_model=data.cpu_model,
            gpu_model=data.gpu_model,
        )

        logger.info(f"[CarbonRunner] Results: {result.emissions_kg:.6f} kg CO2, "
                     f"{result.energy_kwh:.6f} kWh, {result.duration_s:.1f}s")

        return result
