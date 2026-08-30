"""A small custom tool, kept as the worked example of `tools=[...]`.

Uses Open-Meteo, which needs no API key, so the wiring can be exercised without
extra credentials.
"""

import requests
from langchain_core.tools import tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 20

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "depositing rime fog", 51: "light drizzle", 53: "moderate drizzle",
    55: "dense drizzle", 61: "slight rain", 63: "moderate rain", 65: "heavy rain",
    71: "slight snow", 73: "moderate snow", 75: "heavy snow", 80: "rain showers",
    81: "moderate rain showers", 82: "violent rain showers", 95: "thunderstorm",
    96: "thunderstorm with hail", 99: "thunderstorm with heavy hail",
}


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Returns temperature, wind, and conditions."""
    try:
        geo = requests.get(GEOCODE_URL, params={"name": city, "count": 1}, timeout=TIMEOUT)
        geo.raise_for_status()
        matches = geo.json().get("results") or []
        if not matches:
            return f"No location found for '{city}'."

        place = matches[0]
        forecast = requests.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            },
            timeout=TIMEOUT,
        )
        forecast.raise_for_status()
        current = forecast.json().get("current", {})
    except requests.RequestException as exc:
        return f"Weather lookup failed for '{city}': {exc}"

    label = ", ".join(part for part in (place.get("name"), place.get("country")) if part)
    condition = WEATHER_CODES.get(current.get("weather_code"), "unknown conditions")
    return (
        f"{label}: {current.get('temperature_2m')}°C, {condition}, "
        f"humidity {current.get('relative_humidity_2m')}%, "
        f"wind {current.get('wind_speed_10m')} km/h."
    )
