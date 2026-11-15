# {{name}}

A client for a JSON web API, standard library only. Every request has a timeout,
retries are counted and spaced out, and a failure says what was expected and what
arrived.

    from {{name}} import ApiClient, ApiError

    client = ApiClient("https://api.example.com", timeout=5, retries=2)

    try:
        tasks = client.get("/tasks")
    except ApiError as problem:
        print(problem)   # GET https://api.example.com/tasks: expected status 200, got 503 ...

`client.post("/tasks", {"title": "Task"})` sends JSON. Only network faults and 5xx
replies are retried; a 4xx is raised at once, because a second try would fail the
same way. `backoff` is the base delay in seconds, growing with each attempt.

Point the client at the real base URL and add the routes the project uses.

Test with:

    python3 -m pytest -q

The tests start a local HTTP server on a free port, so they need no network.
