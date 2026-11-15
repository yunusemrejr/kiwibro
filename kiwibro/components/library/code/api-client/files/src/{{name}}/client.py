"""A JSON API client that always ends: timeout per try, bounded retries, clear errors."""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any


class ApiError(Exception):
    """A request that failed, saying what was expected and what arrived."""


class ApiClient:
    """Calls a JSON API. Retries only where a second try can help: network faults and 5xx."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 5.0,
        retries: int = 2,
        backoff: float = 0.5,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.backoff = backoff

    def get(self, path: str) -> Any:
        return self._request("GET", path)

    def post(self, path: str, payload: dict) -> Any:
        return self._request("POST", path, payload)

    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {"accept": "application/json"}
        if body is not None:
            headers["content-type"] = "application/json"
        attempts = self.retries + 1
        problem = ""
        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return self._decode(method, url, response)
            except urllib.error.HTTPError as exc:
                problem = f"{method} {url}: expected status 200, got {exc.code} {exc.reason}"
                if exc.code < 500:
                    raise ApiError(problem) from exc  # a 4xx repeats itself next time
            except (urllib.error.URLError, TimeoutError) as exc:
                reason = getattr(exc, "reason", exc)
                problem = f"{method} {url}: expected an answer within {self.timeout}s, got {reason}"
            if attempt < attempts:
                time.sleep(self.backoff * attempt)
        plural = "" if attempts == 1 else "s"
        raise ApiError(f"{problem} (gave up after {attempts} attempt{plural})")

    def _decode(self, method: str, url: str, response) -> Any:
        raw = response.read()
        kind = response.headers.get_content_type()
        if kind != "application/json":
            raise ApiError(f"{method} {url}: expected JSON, got {kind} with {len(raw)} bytes")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ApiError(f"{method} {url}: expected valid JSON, got {raw[:60]!r}") from exc
