FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Le CSV UCI doit être téléchargé localement avant la construction (voir README).
COPY data/log2.csv data/log2.csv
COPY train.py features.py ./
RUN python train.py

COPY api.py app.py ./
COPY tests/ tests/
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000 8501
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
