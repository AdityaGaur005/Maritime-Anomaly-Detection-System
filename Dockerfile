# Use a slim Python image
FROM python:3.11-slim

# Non-root user for security
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Copy requirements first (layer caching: only re-runs pip if requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (files excluded via .dockerignore)
COPY . .

# Hand ownership to the non-root user
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe: hit /health every 30s; unhealthy after 3 failures
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Single worker by default. For multi-worker deployments you need a shared Redis
# store for per-vessel buffers — see mlops/feature_factory.py module docstring.
CMD ["uvicorn", "mlops.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]