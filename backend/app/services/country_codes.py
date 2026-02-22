"""Mapping from ISO 3166-1 alpha-3 codes to Electricity Maps zone identifiers."""

# Electricity Maps uses 2-letter ISO codes or sub-national zones (e.g. "US-MIDA-PJM").
# This covers all countries in our CARBON_INTENSITY dict plus major cloud locations.

ISO3_TO_ZONE: dict[str, str] = {
    # EU
    "AUT": "AT", "BEL": "BE", "BGR": "BG", "HRV": "HR", "CYP": "CY",
    "CZE": "CZ", "DNK": "DK-DK1", "EST": "EE", "FIN": "FI", "FRA": "FR",
    "DEU": "DE", "GRC": "GR", "HUN": "HU", "IRL": "IE", "ITA": "IT-NO",
    "LVA": "LV", "LTU": "LT", "LUX": "LU", "MLT": "MT", "NLD": "NL",
    "POL": "PL", "PRT": "PT", "ROU": "RO", "SVK": "SK", "SVN": "SI",
    "ESP": "ES", "SWE": "SE", "GBR": "GB",
    # Major non-EU
    "USA": "US-MIDA-PJM", "CAN": "CA-QC", "BRA": "BR-S", "CHN": "CN",
    "IND": "IN-NO", "JPN": "JP-TK", "KOR": "KR", "AUS": "AU-NSW",
    "RUS": "RU", "ZAF": "ZA", "CHE": "CH", "NOR": "NO-NO1", "ISR": "IL",
    "SGP": "SG", "IDN": "ID",
}


def iso3_to_zone(iso3: str) -> str | None:
    """Convert a 3-letter ISO code to an Electricity Maps zone. Returns None if unknown."""
    return ISO3_TO_ZONE.get(iso3.upper())
