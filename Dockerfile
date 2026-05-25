FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for typical image processing if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Ensure the model directory exists
RUN mkdir -p model

# Expose the standard port
EXPOSE 8000

# Run the FastAPI application using uvicorn
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
