import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Any  # noqa: E402

import pytest  # noqa: E402

from kiwibro import weather  # noqa: E402

PLACE_CASES = [
    ("whats the weather in istanbul", "istanbul"),
    ("what's the weather like in New York today?", "New York"),
    ("weather in san francisco tomorrow", "san francisco"),
    ("is it raining in london right now", "london"),
    ("will it rain in Izmir this weekend?", "Izmir"),
    ("temperature in cape town", "cape town"),
    ("how hot is it in Dubai", "Dubai"),
    ("how cold is it in oslo today", "oslo"),
    ("weather forecast for Lisbon", "Lisbon"),
    ("do i need an umbrella in paris tomorrow", "paris"),
    ("tokyo weather", "tokyo"),
]

NOT_WEATHER_CASES = [
    "whats the weather",
    "what is weather",
    "how does weather forecasting work",
    "write a poem about the weather in spring",
    "the weather was nice when i was in rome last year",
    "what's the capital of france",
    "whether to use redis or postgres in production",
    "explain climate change in africa",
    "is it cold in here",
    "weather in my app shows the wrong icon, how do i fix the css",
    "cold war in europe",
]


def _forecast_fake(geocoding: Any, forecast: Any) -> Any:
    def fake(url: str, params: dict[str, Any]) -> Any:
        if "geocoding" in url:
            return geocoding
        return forecast

    return fake


LISBON_GEOCODING = {"results": [{"name": "Lisbon", "country": "Portugal", "admin1": "Lisbon",
                                 "latitude": 38.72, "longitude": -9.14}]}
LISBON_FORECAST = {
    "current": {"time": "2026-11-11T19:15", "temperature_2m": 12.6, "apparent_temperature": 10.9,
                "relative_humidity_2m": 78, "precipitation": 0.0, "weather_code": 3, "wind_speed_10m": 17.4},
    "daily": {"time": ["2026-11-11", "2026-11-12", "2026-11-13"],
              "weather_code": [3, 61, 0],
              "temperature_2m_max": [14.2, 11.8, 9.9],
              "temperature_2m_min": [8.1, 7.4, 2.0],
              "precipitation_probability_max": [20, 80, None]},
}
LISBON_FOUND = {
    "place": "Lisbon, Portugal",
    "time": "2026-11-11T19:15",
    "temperature": 12.6,
    "feels_like": 10.9,
    "humidity": 78,
    "wind": 17.4,
    "sky": "overcast",
    "days": [
        {"date": "2026-11-11", "low": 8.1, "high": 14.2, "sky": "overcast", "rain_chance": 20},
        {"date": "2026-11-12", "low": 7.4, "high": 11.8, "sky": "light rain", "rain_chance": 80},
        {"date": "2026-11-13", "low": 2.0, "high": 9.9, "sky": "clear sky", "rain_chance": None},
    ],
}
LISBON_TEXT = """Lisbon, Portugal right now: 13 °C (55 °F), overcast, feels like 11 °C. Wind 17 km/h, humidity 78%.

- Today: 8 to 14 °C, overcast, 20% chance of rain
- Tomorrow: 7 to 12 °C, light rain, 80% chance of rain
- Friday 13 November: 2 to 10 °C, clear sky

Source: Open-Meteo, as of 19:15 local time there."""


def _no_call(url: str, params: dict[str, Any]) -> Any:
    raise AssertionError("the weather service must not be called")


@pytest.mark.parametrize(("message", "place"), PLACE_CASES)
def test_place_asked_finds_the_place(message: str, place: str) -> None:
    assert weather.place_asked(message) == place


@pytest.mark.parametrize("message", NOT_WEATHER_CASES)
def test_place_asked_is_empty_for_other_messages(message: str) -> None:
    assert weather.place_asked(message) == ""


def test_look_reads_geocoding_and_forecast() -> None:
    found = weather.look("Lisbon", _forecast_fake(LISBON_GEOCODING, LISBON_FORECAST))
    assert found == LISBON_FOUND


def test_look_raises_for_an_empty_geocoding_answer() -> None:
    with pytest.raises(weather.WeatherError):
        weather.look("Nowhereville", _forecast_fake({"results": []}, LISBON_FORECAST))


def test_look_raises_when_the_forecast_has_no_current() -> None:
    broken = {"daily": LISBON_FORECAST["daily"]}
    with pytest.raises(weather.WeatherError):
        weather.look("Lisbon", _forecast_fake(LISBON_GEOCODING, broken))


def test_told_gives_the_exact_text() -> None:
    assert weather.told(LISBON_FOUND) == LISBON_TEXT


def test_told_leaves_out_empty_sky_and_missing_rain_chance() -> None:
    found = {**LISBON_FOUND, "sky": "", "days": [
        {"date": "2026-11-11", "low": 8.1, "high": 14.2, "sky": "", "rain_chance": None},
    ]}
    assert weather.told(found) == (
        "Lisbon, Portugal right now: 13 °C (55 °F), feels like 11 °C. Wind 17 km/h, humidity 78%.\n"
        "\n"
        "- Today: 8 to 14 °C\n"
        "\n"
        "Source: Open-Meteo, as of 19:15 local time there."
    )


def test_answer_is_empty_for_a_non_weather_message_without_calling_the_service() -> None:
    assert weather.answer("what's the capital of france", _no_call) == ""


def test_answer_is_empty_when_the_service_fails() -> None:
    def failing(url: str, params: dict[str, Any]) -> Any:
        raise weather.WeatherError("the weather service could not be reached")

    assert weather.answer("weather in Lisbon", failing) == ""


def test_answer_writes_the_text_from_the_service() -> None:
    fake = _forecast_fake(LISBON_GEOCODING, LISBON_FORECAST)
    assert weather.answer("whats the weather in Lisbon", fake) == LISBON_TEXT


@pytest.mark.parametrize("message", [
    "best hot dogs in chicago", "a hot topic in germany", "the cold war in europe", "what's hot in fashion today",
])
def test_hot_and_cold_are_weather_only_as_a_question_about_the_air(message: str) -> None:
    assert weather.place_asked(message) == ""


def test_a_place_the_service_does_not_know_by_that_name_is_no_answer() -> None:
    def get(url: str, params: dict) -> dict:
        assert "geocoding" in url  # the forecast is never asked for
        return {"results": [{"name": "Travelers Rest", "country": "United States", "latitude": 34.9, "longitude": -82.4}]}

    with pytest.raises(weather.WeatherError):
        weather.look("travel", get)
    assert weather.answer("best umbrella for travel", get) == ""
    assert weather._same_place("são paulo", "Sao Paulo") and weather._same_place("hague", "The Hague")


def test_no_weather_request_leaves_the_machine_in_tests() -> None:
    assert weather.answer("whats the weather in Lisbon") == ""  # conftest switches the providers off
