import os
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from app.core.config import settings

# Add vendored codecarbon to path
CODECARBON_PATH = str(settings.PROJECT_ROOT / "codecarbon")
if CODECARBON_PATH not in sys.path:
    sys.path.insert(0, CODECARBON_PATH)

from codecarbon import OfflineEmissionsTracker  # noqa: E402


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
            # Use Popen with process group so we can kill the whole tree on timeout
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
            if proc.returncode != 0:
                raise RuntimeError(
                    f"Training command failed (exit {proc.returncode}):\n"
                    f"STDOUT: {stdout[-2000:]}\n"
                    f"STDERR: {stderr[-2000:]}"
                )
        except subprocess.TimeoutExpired:
            # Kill entire process group
            if proc:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait()
        finally:
            tracker.stop()

        data = tracker.final_emissions_data

        return EmissionsResult(
            emissions_kg=data.emissions,
            energy_kwh=data.energy_consumed,
            duration_s=data.duration,
            cpu_energy=data.cpu_energy,
            gpu_energy=data.gpu_energy,
            water_l=data.water_consumed,
            cpu_model=data.cpu_model,
            gpu_model=data.gpu_model,
        )
