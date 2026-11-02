FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for numpy, matplotlib)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directory for saved models
RUN mkdir -p /app/model/saved_models

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:7860/ || exit 1

# Run the application
CMD ["python", "app/main.py"]
