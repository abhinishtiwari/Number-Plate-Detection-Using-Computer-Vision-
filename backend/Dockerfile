FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies including Tesseract OCR & OpenCV GL libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose default port (Render dynamically sets $PORT environment variable)
ENV PORT=8000
EXPOSE $PORT

# Start application using uvicorn binding to 0.0.0.0:$PORT
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT}"
