FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/processed /app/logs

# Set environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port for API
EXPOSE 8000

# Run application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]