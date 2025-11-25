FROM python:3.10-slim


WORKDIR /app

# Install system dependencies (git required for pip install from GitHub)
RUN apt-get update && apt-get install -y \
    gcc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model code
COPY model.py .
COPY _wsgi.py .

# Set environment variables
ENV ML_BACKEND_PORT=9099
ENV ML_BACKEND_HOST=0.0.0.0
ENV INFERENCE_API_URL=https://aspire.ap.gov.in/inference/inference

# Expose port
EXPOSE 9099

# Run the backend
CMD ["python", "_wsgi.py"]

