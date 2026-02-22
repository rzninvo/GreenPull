"""Wrapper around the Electricity Maps API v3 with fallback to static data."""

import csv
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from app.core.config import settings
from app.services.country_codes import iso3_to_zone
from app.services.green_estimator import CARBON_INTENSITY

logger = logging.getLogger("greenpull")

API_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CarbonIntensityResult:
    zone: str
    carbon_intensity: float  # gCO2/kWh
    datetime_utc: str
    is_estimated: bool
    source: str  # "api" or "static_fallback"


@dataclass
class ForecastWindow:
    datetime_utc: str
    carbon_intensity: float  # gCO2/kWh


@dataclass
class GreenWindowResult:
    best_window_start: str
    best_window_end: str
    best_intensity: float  # gCO2/kWh at optimal time
    current_intensity: float  # gCO2/kWh right now
    forecast: list[ForecastWindow]
    additional_savings_pct: float


@dataclass
class CloudRegion:
    provider: str
    region_code: str
    region_name: str
    country_name: str
    country_iso3: str
    city: str
    carbon_intensity: float  # gCO2/kWh


@dataclass
class RegionRecommendation:
    current_zone: str
    current_intensity: float
    recommended: CloudRegion
    savings_pct: float


@dataclass
class WaterUsageResult:
    water_liters_baseline: float
    water_liters_optimized: float
    wue: float  # L/kWh used


# ---------------------------------------------------------------------------
# Cloud region data loader (from CodeCarbon impact.csv)
# ---------------------------------------------------------------------------

_CLOUD_REGIONS: list[CloudRegion] | None = None


def _load_cloud_regions() -> list[CloudRegion]:
    global _CLOUD_REGIONS
    if _CLOUD_REGIONS is not None:
        return _CLOUD_REGIONS

    csv_path = (
        settings.PROJECT_ROOT
        / "codecarbon" / "codecarbon" / "data" / "cloud" / "impact.csv"
    )
    regions: list[CloudRegion] = []
    if csv_path.exists():
        with open(csv_path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    regions.append(CloudRegion(
                        provider=row["provider"],
                        region_code=row["region"],
                        region_name=row.get("regionName", ""),
                        country_name=row.get("country_name", ""),
                        country_iso3=row.get("countryIsoCode", ""),
                        city=row.get("city", ""),
                        carbon_intensity=float(row["impact"]),
                    ))
                except (ValueError, KeyError) as e:
                    logger.warning(f"Skipping malformed row in impact.csv: {e}")
    else:
        logger.warning(f"Cloud impact CSV not found at {csv_path}")

    _CLOUD_REGIONS = regions
    return regions


# ---------------------------------------------------------------------------
# Main service
# ---------------------------------------------------------------------------


class ElectricityMapsService:
    """Fetches real-time and forecast carbon intensity from Electricity Maps API."""

    def __init__(self, api_token: str | None = None):
        self.api_token = api_token or settings.ELECTRICITYMAPS_API_TOKEN
        self.base_url = settings.ELECTRICITYMAPS_BASE_URL

    def _headers(self) -> dict:
        return {"auth-token": self.api_token}

    def _get(self, endpoint: str, params: dict) -> Optional[dict]:
        """Make a GET request to the API. Returns None on failure."""
        url = f"{self.base_url}{endpoint}"
        try:
            resp = requests.get(
                url, params=params, headers=self._headers(), timeout=API_TIMEOUT
            )
            if resp.status_code == 200:
                return resp.json()
            logger.warning(
                f"Electricity Maps API {endpoint} returned {resp.status_code}: "
                f"{resp.text[:200]}"
            )
        except requests.RequestException as e:
            logger.warning(f"Electricity Maps API {endpoint} failed: {e}")
        return None

    # --- 1. Real-time carbon intensity ---

    def get_carbon_intensity(self, country_iso3: str) -> CarbonIntensityResult:
        """
        Get current carbon intensity for a country.
        Falls back to static CARBON_INTENSITY dict if the API call fails.
        """
        zone = iso3_to_zone(country_iso3)

        if zone and self.api_token:
            data = self._get("/carbon-intensity/latest", {"zone": zone})
            if data and "carbonIntensity" in data:
                return CarbonIntensityResult(
                    zone=data.get("zone", zone),
                    carbon_intensity=data["carbonIntensity"],
                    datetime_utc=data.get("datetime", ""),
                    is_estimated=data.get("isEstimated", True),
                    source="api",
                )

        # Fallback to static dict
        ci = CARBON_INTENSITY.get(country_iso3.upper(), CARBON_INTENSITY["WORLD"])
        return CarbonIntensityResult(
            zone=zone or country_iso3,
            carbon_intensity=ci,
            datetime_utc="",
            is_estimated=True,
            source="static_fallback",
        )

    # --- 2. 72-hour forecast / green window ---

    def get_green_window(
        self, country_iso3: str, training_hours: float = 1.0
    ) -> Optional[GreenWindowResult]:
        """
        Query the 72h forecast and find the lowest-intensity window
        long enough for the training duration.
        Returns None if the API call fails.
        """
        zone = iso3_to_zone(country_iso3)
        if not zone or not self.api_token:
            return None

        data = self._get(
            "/carbon-intensity/forecast",
            {"zone": zone, "horizonHours": 72},
        )
        if not data or "forecast" not in data:
            return None

        forecast_points = data["forecast"]
        if len(forecast_points) < 2:
            return None

        forecast = [
            ForecastWindow(
                datetime_utc=pt["datetime"],
                carbon_intensity=pt["carbonIntensity"],
            )
            for pt in forecast_points
            if "carbonIntensity" in pt and "datetime" in pt
        ]

        if not forecast:
            return None

        current_intensity = forecast[0].carbon_intensity

        # Sliding window: find the window with minimum average intensity
        window_size = max(1, int(training_hours))
        best_avg = float("inf")
        best_start_idx = 0

        for i in range(len(forecast) - window_size + 1):
            window_avg = sum(
                f.carbon_intensity for f in forecast[i : i + window_size]
            ) / window_size
            if window_avg < best_avg:
                best_avg = window_avg
                best_start_idx = i

        best_start = forecast[best_start_idx].datetime_utc
        best_end_idx = min(best_start_idx + window_size - 1, len(forecast) - 1)
        best_end = forecast[best_end_idx].datetime_utc

        savings_pct = (
            (current_intensity - best_avg) / current_intensity * 100
            if current_intensity > 0
            else 0.0
        )

        return GreenWindowResult(
            best_window_start=best_start,
            best_window_end=best_end,
            best_intensity=round(best_avg, 1),
            current_intensity=round(current_intensity, 1),
            forecast=forecast,
            additional_savings_pct=round(max(0, savings_pct), 1),
        )

    # --- 3. Region recommendation ---

    def get_region_recommendation(
        self, country_iso3: str, current_ci: Optional[CarbonIntensityResult] = None
    ) -> Optional[RegionRecommendation]:
        """
        Compare the user's current zone to all known cloud regions
        and recommend the greenest one.
        Pass current_ci to avoid a duplicate API call.
        """
        current = current_ci or self.get_carbon_intensity(country_iso3)

        regions = _load_cloud_regions()
        if not regions:
            return None

        greenest = min(regions, key=lambda r: r.carbon_intensity)

        savings_pct = (
            (current.carbon_intensity - greenest.carbon_intensity)
            / current.carbon_intensity
            * 100
            if current.carbon_intensity > 0
            else 0.0
        )

        return RegionRecommendation(
            current_zone=current.zone,
            current_intensity=current.carbon_intensity,
            recommended=greenest,
            savings_pct=round(max(0, savings_pct), 1),
        )

    # --- 4. Water usage estimation ---

    @staticmethod
    def estimate_water_usage(
        baseline_energy_kwh: float,
        optimized_energy_kwh: float,
        wue: float | None = None,
    ) -> WaterUsageResult:
        """
        Estimate water usage using WUE (Water Usage Effectiveness).
        Formula: water_liters = WUE * energy_kWh
        Follows the CodeCarbon pattern.
        """
        effective_wue = wue if wue is not None else settings.DEFAULT_WUE
        return WaterUsageResult(
            water_liters_baseline=round(effective_wue * baseline_energy_kwh, 4),
            water_liters_optimized=round(effective_wue * optimized_energy_kwh, 4),
            wue=effective_wue,
        )
