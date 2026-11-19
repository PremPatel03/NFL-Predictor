"""
Fetches kickoff-hour stadium weather forecasts from the free Open-Meteo API
for outdoor NFL games (temperature in °F, wind speed in mph, precipitation probability).
"""

import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Stadium coordinates (lat, lon) and default roof type by home team abbreviation
NFL_STADIUMS: dict[str, tuple[float, float, str]] = {
    "ARI": (33.5276, -112.2626, "retractable"),
    "ATL": (33.7554, -84.4008, "retractable"),
    "BAL": (39.2780, -76.6227, "outdoors"),
    "BUF": (42.7738, -78.7870, "outdoors"),
    "CAR": (35.2258, -80.8528, "outdoors"),
    "CHI": (41.8623, -87.6167, "outdoors"),
    "CIN": (39.0955, -84.5161, "outdoors"),
    "CLE": (41.5061, -81.6995, "outdoors"),
    "DAL": (32.7473, -97.0945, "retractable"),
    "DEN": (39.7439, -105.0201, "outdoors"),
    "DET": (42.3400, -83.0456, "dome"),
    "GB":  (44.5013, -88.0622, "outdoors"),
    "HOU": (29.6847, -95.4107, "retractable"),
    "IND": (39.7601, -86.1639, "retractable"),
    "JAX": (30.3239, -81.6373, "outdoors"),
    "KC":  (39.0489, -94.4839, "outdoors"),
    "LA":  (33.9535, -118.3390, "dome"),
    "LAC": (33.9535, -118.3390, "dome"),
    "LV":  (36.0909, -115.1833, "dome"),
    "MIA": (25.9580, -80.2389, "outdoors"),
    "MIN": (44.9736, -93.2575, "dome"),
    "NE":  (42.0909, -71.2643, "outdoors"),
    "NO":  (29.9511, -90.0812, "dome"),
    "NYG": (40.8135, -74.0745, "outdoors"),
    "NYJ": (40.8135, -74.0745, "outdoors"),
    "PHI": (39.9008, -75.1675, "outdoors"),
    "PIT": (40.4468, -80.0158, "outdoors"),
    "SEA": (47.5952, -122.3316, "outdoors"),
    "SF":  (37.4033, -121.9694, "outdoors"),
    "TB":  (27.9759, -82.5033, "outdoors"),
    "TEN": (36.1665, -86.7713, "outdoors"),
    "WAS": (38.9076, -76.8645, "outdoors"),
}

def get_game_weather(home_team: str, gameday: str, gametime_24h: str, roof: str | None = None) -> dict:
    """
    Returns weather summary dict (`weather_summary`, `temp_f`, `wind_mph`, `precip_prob`, `weather_alert`)
    for a given game date/time in America/New_York timezone.
    """
    info = NFL_STADIUMS.get(home_team)
    if not info:
        return {"weather_summary": "N/A", "temp_f": None, "wind_mph": None, "precip_prob": None, "weather_alert": "Normal"}

    lat, lon, default_roof = info
    effective_roof = str(roof).lower() if pd.notna(roof) and str(roof).strip() else default_roof

    if effective_roof in {"dome", "closed"}:
        return {
            "weather_summary": "Dome / Indoor (70°F)",
            "temp_f": 70.0,
            "wind_mph": 0.0,
            "precip_prob": 0,
            "weather_alert": "Indoor",
        }

    try:
        kickoff_hour = int(str(gametime_24h).split(":")[0]) if ":" in str(gametime_24h) else 13
        target_ts = f"{gameday}T{kickoff_hour:02d}:00"

        params = {
            "latitude": lat,
            "longitude": lon,
            "hourly": "temperature_2m,wind_speed_10m,precipitation_probability",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/New_York",
            "start_date": gameday,
            "end_date": gameday,
        }
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=10).json()
        hourly = resp.get("hourly", {})
        times = hourly.get("time", [])

        idx = times.index(target_ts) if target_ts in times else min(kickoff_hour, len(times) - 1)
        temp_f = round(float(hourly["temperature_2m"][idx]))
        wind_mph = round(float(hourly["wind_speed_10m"][idx]))
        precip = int(hourly["precipitation_probability"][idx] or 0)

        alert = "Normal"
        if wind_mph >= 15:
            alert = f"High Wind ({wind_mph} mph)"
        elif precip >= 50:
            alert = f"Rain/Snow ({precip}%)"
        elif temp_f <= 32:
            alert = f"Freezing ({temp_f}°F)"

        summary = f"{temp_f}°F • Wind {wind_mph} mph"
        if precip >= 30:
            summary += f" • Precip {precip}%"

        return {
            "weather_summary": summary,
            "temp_f": temp_f,
            "wind_mph": wind_mph,
            "precip_prob": precip,
            "weather_alert": alert,
        }
    except Exception:
        return {"weather_summary": "Forecast Pending", "temp_f": None, "wind_mph": None, "precip_prob": None, "weather_alert": "Normal"}
