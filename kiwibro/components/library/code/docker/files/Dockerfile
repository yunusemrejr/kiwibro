FROM python:3.12-slim

# Unbuffered output reaches docker logs at once; no bytecode keeps the image clean.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# requirements.txt first, so a code change does not repeat the install layer.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

# A fixed unprivileged account; the app never writes to its own code.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8000
CMD ["python", "app.py"]
