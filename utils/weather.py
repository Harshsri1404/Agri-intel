"""
AgriIntel — Weather API + NPA Early Warning System
Uses OpenWeatherMap (free tier) for drought/flood detection
"""

import requests
import os
from datetime import datetime

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "YOUR_FREE_KEY_FROM_OPENWEATHERMAP")
BASE_URL = "https://api.openweathermap.org/data/2.5"


def get_district_weather(district: str, state: str = "India") -> dict:
    """Fetch current weather for a district."""
    try:
        url = f"{BASE_URL}/weather"
        params = {
            "q": f"{district},{state},IN",
            "appid": WEATHER_API_KEY,
            "units": "metric"
        }
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            return {
                "district":    district,
                "temp":        data["main"]["temp"],
                "humidity":    data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "rainfall_mm": data.get("rain", {}).get("1h", 0),
                "wind_speed":  data["wind"]["speed"],
                "fetched_at":  datetime.now().isoformat()
            }
    except Exception:
        pass
    # Fallback: return neutral data
    return {
        "district": district, "temp": 28, "humidity": 60,
        "description": "data unavailable", "rainfall_mm": 0,
        "wind_speed": 5, "fetched_at": datetime.now().isoformat()
    }


def assess_npa_risk(weather: dict, crop_type: str = "") -> dict:
    """
    Assess NPA risk from weather conditions.
    Returns risk level and recommended action.
    """
    temp      = weather.get("temp", 28)
    humidity  = weather.get("humidity", 60)
    rainfall  = weather.get("rainfall_mm", 0)
    wind      = weather.get("wind_speed", 5)
    desc      = weather.get("description", "").lower()

    risk_score = 0
    reasons    = []

    # ── Drought indicators
    if humidity < 30:
        risk_score += 3
        reasons.append("Very low humidity — drought conditions")
    elif humidity < 45:
        risk_score += 1
        reasons.append("Low humidity — dry spell likely")

    if temp > 42:
        risk_score += 3
        reasons.append("Extreme heat — crop stress likely")
    elif temp > 38:
        risk_score += 1
        reasons.append("High temperature — monitor crop health")

    # ── Flood indicators
    if rainfall > 50:
        risk_score += 4
        reasons.append("Heavy rainfall — flooding risk high")
    elif rainfall > 20:
        risk_score += 2
        reasons.append("Moderate rainfall — waterlogging possible")

    # ── Storm/cyclone
    if wind > 20:
        risk_score += 3
        reasons.append("High wind speed — crop damage risk")
    elif wind > 12:
        risk_score += 1
        reasons.append("Moderate wind")

    # ── Text-based detection
    for keyword in ["storm","flood","drought","cyclone","hail","thunder"]:
        if keyword in desc:
            risk_score += 3
            reasons.append(f"Weather alert: {desc}")
            break

    # ── Crop-specific modifiers
    sensitive_crops = ["cotton","sugarcane","vegetables","onion","tomato"]
    if crop_type.lower() in sensitive_crops:
        risk_score = int(risk_score * 1.3)

    # ── Risk level
    if risk_score >= 6:
        level  = "HIGH"
        action = "Immediate contact recommended — consider loan restructuring"
        color  = "RED"
    elif risk_score >= 3:
        level  = "MEDIUM"
        action = "Monitor closely — schedule courtesy call with farmer"
        color  = "YELLOW"
    else:
        level  = "LOW"
        action = "No action required — routine monitoring"
        color  = "GREEN"

    return {
        "risk_level":  level,
        "risk_score":  risk_score,
        "risk_color":  color,
        "reasons":     reasons,
        "action":      action,
        "weather":     weather
    }


def get_district_npa_risk(district: str, crop_type: str = "",
                           state: str = "India") -> dict:
    """Combined: fetch weather + assess NPA risk."""
    weather = get_district_weather(district, state)
    risk    = assess_npa_risk(weather, crop_type)
    return risk
