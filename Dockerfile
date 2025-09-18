FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model (if already trained), pipeline code, and serving app
COPY models/ models/
COPY pipeline/ pipeline/
COPY deployment/app/ serve/

# Start FastAPI server
CMD ["uvicorn", "serve.main:app", "--host", "0.0.0.0", "--port", "8080"]
