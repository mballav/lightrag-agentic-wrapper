FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lightrag_wrapper ./lightrag_wrapper

RUN mkdir -p /app/data/rag_storage /app/data/inputs /app/data/wrapper/uploads /app/data/query_dags \
    && useradd --create-home --uid 1000 app \
    && chown -R app:app /app

USER app

EXPOSE 8000

CMD ["python", "-m", "lightrag_wrapper"]
