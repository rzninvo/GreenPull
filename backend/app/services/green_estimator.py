import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("greenpull")

# ---------------------------------------------------------------------------
# Hardware TDP lookup tables
# ---------------------------------------------------------------------------

# GPU TDP in watts (source: NVIDIA spec sheets)
GPU_TDP: dict[str, float] = {
    # Data center GPUs
    "H100": 700, "H100_SXM": 700, "H100_PCIE": 350,
    "A100": 400, "A100_SXM": 400, "A100_PCIE": 300, "A100_80GB": 400,
    "V100": 300, "V100S": 250,
    "A10": 150, "A10G": 150, "A30": 165, "A40": 300,
    "L4": 72, "L40": 300, "L40S": 350,
    "T4": 70, "P100": 250, "K80": 300,
    # Consumer GPUs
    "RTX_4090": 450, "RTX_4080": 320, "RTX_4070": 200,
    "RTX_3090": 350, "RTX_3080": 320, "RTX_3070": 220, "RTX_3060": 170,
    "RTX_2080_TI": 250, "RTX_2080": 215, "RTX_2070": 175,
    "GTX_1080_TI": 250, "GTX_1080": 180,
    # Apple
    "M1_GPU": 20, "M2_GPU": 25, "M3_GPU": 30,
    # Fallback
    "UNKNOWN_GPU": 250,
}

# CPU classes: total TDP and typical core counts
CPU_TDP: dict[str, dict] = {
    "xeon_scalable": {"total_tdp": 205, "cores": 32, "name": "Intel Xeon Scalable"},
    "xeon_w": {"total_tdp": 165, "cores": 16, "name": "Intel Xeon W"},
    "epyc_genoa": {"total_tdp": 360, "cores": 96, "name": "AMD EPYC 9004 (Genoa)"},
    "epyc_milan": {"total_tdp": 280, "cores": 64, "name": "AMD EPYC 7003 (Milan)"},
    "epyc_rome": {"total_tdp": 225, "cores": 64, "name": "AMD EPYC 7002 (Rome)"},
    "ryzen_9": {"total_tdp": 170, "cores": 16, "name": "AMD Ryzen 9"},
    "i9_desktop": {"total_tdp": 125, "cores": 16, "name": "Intel Core i9"},
    "cloud_generic": {"total_tdp": 200, "cores": 32, "name": "Generic Cloud CPU"},
    "server_generic": {"total_tdp": 200, "cores": 32, "name": "Generic Server CPU"},
    "desktop_generic": {"total_tdp": 65, "cores": 8, "name": "Generic Desktop CPU"},
    "UNKNOWN": {"total_tdp": 150, "cores": 16, "name": "Unknown CPU"},
}

# ---------------------------------------------------------------------------
# Carbon intensity by country (gCO2 per kWh)
# EU: codecarbon CSV (2020), Non-EU: IEA / electricityMap estimates
# ---------------------------------------------------------------------------

CARBON_INTENSITY: dict[str, float] = {
    # EU
    "AUT": 83, "BEL": 192, "BGR": 352, "HRV": 164, "CYP": 653,
    "CZE": 386, "DNK": 116, "EST": 669, "FIN": 67, "FRA": 55,
    "DEU": 301, "GRC": 522, "HUN": 218, "IRL": 293, "ITA": 212,
    "LVA": 92, "LTU": 146, "LUX": 69, "MLT": 356, "NLD": 318,
    "POL": 724, "PRT": 201, "ROU": 208, "SVK": 90, "SVN": 219,
    "ESP": 190, "SWE": 13, "GBR": 209,
    # Major non-EU
    "USA": 379, "CAN": 120, "BRA": 56, "CHN": 555, "IND": 632,
    "JPN": 459, "KOR": 378, "AUS": 501, "RUS": 364, "ZAF": 646,
    "CHE": 59, "NOR": 8, "ISR": 463, "SGP": 369, "IDN": 580,
    # World average
    "WORLD": 436,
}

# ---------------------------------------------------------------------------
# Constants for comparison metrics
# ---------------------------------------------------------------------------

MEMORY_POWER_W_PER_GB = 0.375
PUE_LOCAL = 1.0
PUE_CLOUD = 1.2
TREE_KG_CO2_PER_MONTH = 1.83       # ~22 kg/year / 12
CAR_KG_CO2_PER_KM = 0.21           # average European car
SMARTPHONE_CHARGE_KWH = 0.012
STREAMING_KWH_PER_HOUR = 0.036
FLIGHT_PARIS_NYC_KG_CO2 = 1000.0   # round-trip per passenger
LED_BULB_WATTS = 10.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    model_type: str
    model_name: Optional[str]
    parameter_count: float          # millions
    framework: str
    epochs: int
    batch_size: int
    dataset_size_estimate: str
    estimated_runtime_hours: float
    gpu_required: bool
    gpu_type: str
    num_gpus: int
    cpu_type: str
    num_cpu_cores: int
    memory_gb: float
    reasoning: str


@dataclass
class HardwareConfig:
    gpu_tdp_w: float
    gpu_name: str
    cpu_tdp_w: float                # per-core
    cpu_name: str
    num_gpus: int
    num_cpu_cores: int
    memory_gb: float
    pue: float
    gpu_usage_factor: float
    cpu_usage_factor: float


@dataclass
class EstimationResult:
    emissions_kg: float
    energy_kwh: float
    duration_s: float
    power_w: float
    cpu_energy_kwh: float
    gpu_energy_kwh: float
    memory_energy_kwh: float
    carbon_intensity: float
    hardware_config: HardwareConfig
    country_code: str


@dataclass
class ComparisonMetrics:
    tree_months: float
    car_km: float
    smartphone_charges: float
    streaming_hours: float
    flight_fraction: float
    led_bulb_hours: float

# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class GreenEstimator:
    def __init__(self, country_code: str = "USA"):
        self.country_code = country_code.upper()

    def resolve_hardware(self, training_config: TrainingConfig) -> HardwareConfig:
        """Map GPT's hardware description to concrete TDP values."""
        gpu_key = self._normalize_gpu_name(training_config.gpu_type)
        gpu_tdp = GPU_TDP.get(gpu_key, GPU_TDP["UNKNOWN_GPU"])

        cpu_key = self._normalize_cpu_name(training_config.cpu_type)
        cpu_info = CPU_TDP.get(cpu_key, CPU_TDP["UNKNOWN"])
        per_core_tdp = cpu_info["total_tdp"] / cpu_info["cores"]

        is_cloud = "cloud" in training_config.cpu_type.lower()
        pue = PUE_CLOUD if is_cloud else PUE_LOCAL

        # If GPU not required, zero out GPU count
        actual_gpus = max(training_config.num_gpus, 0) if training_config.gpu_required else 0

        hw = HardwareConfig(
            gpu_tdp_w=gpu_tdp,
            gpu_name=gpu_key.replace("_", " ") if actual_gpus > 0 else "None",
            cpu_tdp_w=per_core_tdp,
            cpu_name=cpu_info["name"],
            num_gpus=actual_gpus,
            num_cpu_cores=max(training_config.num_cpu_cores, 1),
            memory_gb=max(training_config.memory_gb, 4.0),
            pue=pue,
            gpu_usage_factor=0.80,
            cpu_usage_factor=0.80 if actual_gpus == 0 else 0.50,  # CPU works harder without GPU
        )

        logger.info(f"[HW] GPU: {hw.gpu_name} ({gpu_tdp}W) x{hw.num_gpus}")
        logger.info(f"[HW] CPU: {hw.cpu_name} ({per_core_tdp:.1f}W/core) x{hw.num_cpu_cores} cores")
        logger.info(f"[HW] Memory: {hw.memory_gb:.0f} GB, PUE: {hw.pue}")
        return hw

    def estimate_emissions(
        self,
        training_config: TrainingConfig,
        hardware_config: Optional[HardwareConfig] = None,
    ) -> EstimationResult:
        """
        Green Algorithms formula:
          power  = PUE * (cpu_cores * TDP/core * usage + GPUs * GPU_TDP * usage + mem_GB * 0.375)
          energy = power * hours / 1000   (kWh)
          carbon = energy * CI(country)   (gCO2)
        """
        hw = hardware_config or self.resolve_hardware(training_config)
        hours = training_config.estimated_runtime_hours

        cpu_power = hw.num_cpu_cores * hw.cpu_tdp_w * hw.cpu_usage_factor
        gpu_power = hw.num_gpus * hw.gpu_tdp_w * hw.gpu_usage_factor
        mem_power = hw.memory_gb * MEMORY_POWER_W_PER_GB
        total_power = hw.pue * (cpu_power + gpu_power + mem_power)

        total_energy = total_power * hours / 1000.0
        cpu_energy = hw.pue * cpu_power * hours / 1000.0
        gpu_energy = hw.pue * gpu_power * hours / 1000.0
        mem_energy = hw.pue * mem_power * hours / 1000.0

        ci = CARBON_INTENSITY.get(self.country_code, CARBON_INTENSITY["WORLD"])
        carbon_kg = total_energy * ci / 1000.0

        logger.info(f"[Estimate] Power: {total_power:.1f}W "
                    f"(CPU={cpu_power:.1f}W GPU={gpu_power:.1f}W Mem={mem_power:.1f}W)")
        logger.info(f"[Estimate] Energy: {total_energy:.4f} kWh over {hours:.2f}h")
        logger.info(f"[Estimate] Carbon: {carbon_kg:.6f} kg CO2 "
                    f"(CI={ci} gCO2/kWh, country={self.country_code})")

        return EstimationResult(
            emissions_kg=carbon_kg,
            energy_kwh=total_energy,
            duration_s=hours * 3600,
            power_w=total_power,
            cpu_energy_kwh=cpu_energy,
            gpu_energy_kwh=gpu_energy,
            memory_energy_kwh=mem_energy,
            carbon_intensity=ci,
            hardware_config=hw,
            country_code=self.country_code,
        )

    def compute_comparisons(self, saved_kg_co2: float, saved_kwh: float) -> ComparisonMetrics:
        """Convert savings into real-world equivalents."""
        return ComparisonMetrics(
            tree_months=saved_kg_co2 / TREE_KG_CO2_PER_MONTH if TREE_KG_CO2_PER_MONTH > 0 else 0,
            car_km=saved_kg_co2 / CAR_KG_CO2_PER_KM if CAR_KG_CO2_PER_KM > 0 else 0,
            smartphone_charges=saved_kwh / SMARTPHONE_CHARGE_KWH if SMARTPHONE_CHARGE_KWH > 0 else 0,
            streaming_hours=saved_kwh / STREAMING_KWH_PER_HOUR if STREAMING_KWH_PER_HOUR > 0 else 0,
            flight_fraction=saved_kg_co2 / FLIGHT_PARIS_NYC_KG_CO2 if FLIGHT_PARIS_NYC_KG_CO2 > 0 else 0,
            led_bulb_hours=saved_kwh * 1000 / LED_BULB_WATTS if LED_BULB_WATTS > 0 else 0,
        )

    # --- Name normalization helpers ---

    @staticmethod
    def _normalize_gpu_name(raw: str) -> str:
        if not raw:
            return "UNKNOWN_GPU"
        name = raw.upper().strip().replace("-", "_").replace(" ", "_")
        name = name.replace("NVIDIA_", "").replace("GEFORCE_", "")
        if name in GPU_TDP:
            return name
        for key in GPU_TDP:
            if key in name or name in key:
                return key
        return "UNKNOWN_GPU"

    @staticmethod
    def _normalize_cpu_name(raw: str) -> str:
        if not raw:
            return "UNKNOWN"
        name = raw.lower().strip()
        if "cloud" in name:
            return "cloud_generic"
        if "epyc" in name and ("genoa" in name or "9" in name):
            return "epyc_genoa"
        if "epyc" in name and "milan" in name:
            return "epyc_milan"
        if "epyc" in name:
            return "epyc_rome"
        if "xeon" in name and any(w in name for w in ("scalable", "gold", "platinum")):
            return "xeon_scalable"
        if "xeon" in name:
            return "xeon_w"
        if "ryzen" in name and "9" in name:
            return "ryzen_9"
        if "i9" in name or "core" in name:
            return "i9_desktop"
        if "server" in name:
            return "server_generic"
        if "desktop" in name:
            return "desktop_generic"
        return "UNKNOWN"
