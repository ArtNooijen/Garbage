FROM ghcr.io/astral-sh/uv:0.9.5 AS uv

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app
ARG INSTALL_YOLO=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv
COPY README.md pyproject.toml uv.lock ./
COPY src ./src

RUN if [ "$INSTALL_YOLO" = "true" ]; then \
        uv sync --frozen --no-dev --extra yolo; \
    else \
        uv sync --frozen --no-dev; \
    fi

VOLUME ["/app/data", "/app/models", "/app/samples"]

ENTRYPOINT ["uv", "run", "--no-sync", "garbage-vision"]
CMD ["prod"]
