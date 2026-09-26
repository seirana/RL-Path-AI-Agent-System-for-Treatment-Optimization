# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY train.py evaluate.py ./
COPY scripts ./scripts

RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.1,<3" \
    && python -m pip install .

RUN mkdir -p /app/data/raw /app/data/processed /app/artifacts

ENTRYPOINT ["rlpath-train"]
CMD ["--help"]
