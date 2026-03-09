FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY server.py soreal_engine.py ./
COPY static ./static
COPY docker ./docker

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/', timeout=5).read()"

CMD ["sh", "-c", "python docker/wait_for_services.py && uvicorn server:app --host 0.0.0.0 --port 8001"]
