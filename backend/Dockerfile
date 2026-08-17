# Build from the REPOSITORY ROOT, not from backend/:
#   docker build -f backend/Dockerfile -t number-plate-ai .
#
# The old image copied only backend/ and ran `uvicorn main:app`, which crashed
# with "attempted relative import with no known parent package" because main.py
# uses package-relative imports. The app must be started as backend.main:app.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# libGL/libglib are needed by OpenCV's wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# The RTO master CSV lives at the repository root, so copy the whole project.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY India_RTO_Registration_Dataset_New.csv ./

# Fail fast at build time if the dataset or OCR engine is missing, rather than
# discovering it on the first request.
RUN python -c "from backend.rto_lookup import rto_engine; \
print('RTO records:', rto_engine.record_count)" \
 && python -c "from backend.ocr_engine import ocr_engine; \
s = ocr_engine.status(); print('OCR:', s['available']); assert s['available'], 'no OCR engine'"

# Render and similar platforms inject $PORT.
ENV PORT=8000
EXPOSE 8000

RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
