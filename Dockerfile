# CPU-oriented baseline. For CUDA, select a matching official PyTorch base image.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# OpenCV headless avoids pulling GUI libraries into the container.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./app.py
RUN pip install --upgrade pip && pip install .

COPY docs ./docs
COPY scripts ./scripts
COPY data ./data
COPY artifacts/.gitkeep ./artifacts/.gitkeep

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
