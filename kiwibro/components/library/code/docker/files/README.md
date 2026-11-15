# {{name}}

A Python web app in a container, plus the compose file that runs it.

    app.py             two JSON routes: / and /health
    requirements.txt   third-party packages; empty for the starter
    Dockerfile         slim Python image, install, then an unprivileged user
    compose.yaml       builds the image, publishes port 8000, checks /health
    .dockerignore      keeps .git, caches, tests and data out of the build context

## Run

    docker compose up --build

`docker compose ps` shows the service as healthy once `/health` answers. The image
runs as `appuser` (uid 10001), so nothing inside the container can write to the
code it runs.

## Change

Add the project's packages to `requirements.txt`; they are installed before the
code is copied, so editing `app.py` rebuilds quickly. The app reads `PORT` from the
environment; keep the compose port mapping and the health check URL in step with it.
