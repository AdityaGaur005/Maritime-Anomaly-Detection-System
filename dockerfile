# Use a slim Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project (excluding files listed in .dockerignore)
COPY . .

# Ensure the container listens on port 8000
EXPOSE 8000

# Run the FastAPI app with uvicorn
# --host 0.0.0.0 makes it accessible from outside the container
CMD ["uvicorn", "mlops.main:app", "--host", "0.0.0.0", "--port", "8000"]