FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# CPU-only torch keeps the image small (no GPU needed for bge-small)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml]"
COPY alembic ./alembic
COPY alembic.ini ./
