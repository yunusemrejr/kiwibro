"""Weather questions answered from real forecast data, not from a model's memory.

A small model asked about the weather invents the numbers, and weather pages cannot be read as
plain text, so Kiwibro reads Open-Meteo's forecast data instead and writes the answer from those
figures. This module finds the place a question names, fetches its current weather and a three-day
forecast, and words the answer. It returns "" when the question is not about weather or the service
cannot answer, so the caller can fall back to a web search.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from typing import Any, Callable

import requests

from . import search

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_S = 8
DAYS = 3
MAX_WORDS = 14  # a weather question is short; a longer message is a request for something else
MAX_PLACE_WORDS = 4

Getter = Callable[[str, dict[str, Any]], dict[str, Any]]

# The standard WMO weather interpretation codes, as Open-Meteo reports them.
WMO: dict[int, str] = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "freezing fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 56: "light freezing drizzle",
    57: "freezing drizzle", 61: "light rain", 63: "rain", 65: "heavy rain", 66: "light freezing rain",
    67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow", 77: "snow grains",
    80: "light rain showers", 81: "rain showers", 82: "heavy rain showers", 85: "light snow showers",
    86: "heavy snow showers", 95: "thunderstorm", 96: "thunderstorm with light hail",
    99: "thunderstorm with hail",
}

# "hot" and "cold" count only as a question about the air ("how hot is it in"), never as "hot dogs in".
_WEATHER_WORDS = (r"weather|forecast|temperature|rain(?:ing)?|snow(?:ing)?|umbrella|"
                  r"(?:how\s+|is\s+it\s+|it\s+be\s+)(?:hot|cold|warm|sunny|windy|humid)")
_FILLER = "is|it|be|like|will|to|forecast|does|feel|look"  # small words allowed between the weather word and "in"
_ASKED = re.compile(
    rf"\b(?:{_WEATHER_WORDS})\b(?:\s+(?:{_FILLER})\b){{0,3}}?\s+(?:in|for|at)\s+(?P<tail>.+)$", re.I)
_SHORT = re.compile(r"^(?P<place>.+?)\s+(?:weather|forecast)[\s?!.]*$", re.I)
_NOT_A_QUESTION = re.compile(
    r"\b(?:was|were|write|poem|story|explain|css|code|app|fix|yesterday)\b|\blast\s+(?:year|week)\b", re.I)
_TIME_WORDS = re.compile(
    r"\b(?:today|tomorrow|tonight|right\s+now|now|this\s+week(?:end)?|next\s+week|"
    r"on\s+(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)|at\s+the\s+moment|currently)\b", re.I)
_CUT_PUNCTUATION = re.compile(r"[?!,;:]|\.(?=\s|$)")
_WORD = re.compile(r"[a-z]+")
_NOT_A_PLACE = frozenset(
    "my our here there spring summer autumn fall winter general production space".split())
_NOT_A_SHORT_PLACE = _NOT_A_PLACE | frozenset(
    "what whats how is the a an it do does will can this that tell me i weather forecast temperature rain raining "
    "snow snowing umbrella hot cold sunny windy humid".split())

_READ_ERROR = "the weather service gave an answer that could not be read"


class WeatherError(Exception):
    """The weather could not be looked up or read."""


def place_asked(message: str) -> str:
    """The place a weather question names, or "" when the message is not one."""
    text = (message or "").replace("’", "'").strip()
    words = text.split()
    if not words or len(words) > MAX_WORDS or _NOT_A_QUESTION.search(text):
        return ""
    asked = _ASKED.search(text)
    if asked:
        return _long_place(asked.group("tail"))
    short = _SHORT.match(text)
    if short and len(words) <= 4:
        return _short_place(short.group("place"))
    return ""


def _long_place(tail: str) -> str:
    """The place after "in"/"for"/"at", up to a time word or punctuation."""
    tail = _TIME_WORDS.split(tail, maxsplit=1)[0]
    tail = _CUT_PUNCTUATION.split(tail, maxsplit=1)[0].strip(" .")
    words = tail.split()
    if not 1 <= len(words) <= MAX_PLACE_WORDS or any(word.lower() in _NOT_A_PLACE for word in words):
        return ""
    return " ".join(words)


def _short_place(place: str) -> str:
    """The place in "<place> weather" or "<place> forecast" (no preposition), when it is a real place."""
    words = place.split()
    if not 1 <= len(words) <= 3 or any(word.lower() in _NOT_A_SHORT_PLACE for word in _WORD.findall(place.lower())):
        return ""
    return " ".join(words)


def _get(url: str, params: dict[str, Any]) -> dict[str, Any]:
    """GET a JSON object from Open-Meteo; WeatherError on any network, status or JSON failure."""
    try:
        response = search._get(url, params=params, timeout=TIMEOUT_S)  # the search providers' session
    except requests.RequestException as exc:
        raise WeatherError("the weather service could not be reached") from exc
    if response.status_code != 200:
        raise WeatherError(f"the weather service answered with status {response.status_code}")
    try:
        data = response.json()
    except ValueError as exc:
        raise WeatherError(_READ_ERROR) from exc
    if not isinstance(data, dict):
        raise WeatherError(_READ_ERROR)
    return data


def look(place: str, get: Getter | None = None) -> dict[str, Any]:
    """The place's current weather and three-day forecast, from Open-Meteo."""
    fetch = get or _get
    found = fetch(GEOCODE_URL, {"name": place, "count": 1, "language": "en", "format": "json"})
    results = found.get("results")
    if not isinstance(results, list) or not results:
        raise WeatherError(f"no place named {place} was found")
    spot = results[0]
    if not isinstance(spot, dict) or "latitude" not in spot or "longitude" not in spot:
        raise WeatherError(_READ_ERROR)
    if not _same_place(place, str(spot.get("name") or "")):  # "umbrella for travel" is not Travelers Rest
        raise WeatherError(f"no place named {place} was found")
    data = fetch(FORECAST_URL, {
        "latitude": spot["latitude"],
        "longitude": spot["longitude"],
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
        "forecast_days": DAYS,
    })
    try:
        return _read(spot, data)
    except (KeyError, TypeError, ValueError) as exc:
        raise WeatherError(_READ_ERROR) from exc


def _fold(name: str) -> str:
    """A place name without accents, case or punctuation, for comparing."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return " ".join(re.findall(r"[a-z0-9]+", plain.lower()))


def _same_place(asked: str, found: str) -> bool:
    """Whether the place the service found is the one that was asked about."""
    asked, found = _fold(asked), _fold(found)
    return bool(asked and found) and (asked == found or asked.removeprefix("the ") == found.removeprefix("the "))


def _read(spot: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    """The look() result, built from the two answers. A missing key or value raises."""
    now = data["current"]
    daily = data["daily"]
    days = [
        {
            "date": day,
            "low": float(low),
            "high": float(high),
            "sky": WMO.get(code, ""),
            "rain_chance": None if rain is None else int(rain),
        }
        for day, low, high, code, rain in zip(
            daily["time"][:DAYS], daily["temperature_2m_min"], daily["temperature_2m_max"],
            daily["weather_code"], daily["precipitation_probability_max"])
    ]
    country = spot.get("country")
    humidity = now["relative_humidity_2m"]
    wind = now["wind_speed_10m"]
    return {
        "place": f"{spot['name']}, {country}" if country else str(spot["name"]),
        "time": now["time"],
        "temperature": float(now["temperature_2m"]),
        "feels_like": float(now["apparent_temperature"]),
        "humidity": None if humidity is None else int(humidity),
        "wind": None if wind is None else float(wind),
        "sky": WMO.get(now["weather_code"], ""),
        "days": days,
    }


def told(found: dict[str, Any]) -> str:
    """The answer text for a look() result: now, one line per day, then the source."""
    sky = f", {found['sky']}" if found["sky"] else ""
    lines = [
        f"{found['place']} right now: {round(found['temperature'])} °C "
        f"({round(found['temperature'] * 9 / 5 + 32)} °F){sky}, feels like {round(found['feels_like'])} °C."
        + _conditions(found["wind"], found["humidity"])
    ]
    if found["days"]:
        lines += ["", *(_day_line(index, day) for index, day in enumerate(found["days"]))]
    clock = found["time"].split("T", 1)[-1][:5]
    lines += ["", f"Source: Open-Meteo, as of {clock} local time there."]
    return "\n".join(lines)


def _conditions(wind: float | None, humidity: int | None) -> str:
    """The wind and humidity sentence; "" when neither could be read."""
    if wind is not None and humidity is not None:
        return f" Wind {round(wind)} km/h, humidity {humidity}%."
    if wind is not None:
        return f" Wind {round(wind)} km/h."
    if humidity is not None:
        return f" Humidity {humidity}%."
    return ""


def _day_line(index: int, day: dict[str, Any]) -> str:
    """One forecast line: today, tomorrow, then the weekday and date."""
    if index == 0:
        label = "Today"
    elif index == 1:
        label = "Tomorrow"
    else:
        when = date.fromisoformat(day["date"])
        label = f"{when:%A} {when.day} {when:%B}"
    sky = f", {day['sky']}" if day["sky"] else ""
    rain = f", {day['rain_chance']}% chance of rain" if day["rain_chance"] is not None else ""
    return f"- {label}: {round(day['low'])} to {round(day['high'])} °C{sky}{rain}"


def answer(message: str, get: Getter | None = None) -> str:
    """The weather answer for a message; "" when it is not a weather question or the service cannot answer."""
    place = place_asked(message)
    if not place:
        return ""
    try:
        return told(look(place, get))
    except WeatherError:
        return ""
