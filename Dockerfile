FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-api.txt .
RUN python -m pip install --no-cache-dir -r requirements-api.txt

COPY api ./api
COPY src ./src
COPY models/catboost_v2.cbm ./models/catboost_v2.cbm
COPY models/model_metadata.json ./models/model_metadata.json

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
