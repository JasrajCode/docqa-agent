FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.9 /uv /uvx /bin/

# Bytecode is compiled at build time so cold starts skip it; the image's Python is used as is.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies change less often than the code, so they get their own cached layer.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked --no-dev --no-install-project

COPY README.md ./
COPY src ./src
RUN uv sync --locked --no-dev

# The app only reads its files, so the non-root user is not given ownership of them.
RUN useradd --uid 1001 --no-create-home app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PORT=8000
EXPOSE 8000

# Render sets PORT; the shell form expands it, and exec makes uvicorn PID 1 so it receives SIGTERM.
CMD ["sh", "-c", "exec uvicorn --factory docqa_agent.api:create_production_app --host 0.0.0.0 --port ${PORT}"]
