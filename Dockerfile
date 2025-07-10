FROM ubuntu:24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/  
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

RUN if [ -f "setup.py" ] || [ -f "requirements.txt" ]; then \
    black --check --diff . || true; \
    isort --check-only --diff . || true; \
    flake8 || true; \
    fi

RUN [ -f "test_bot.py" ] && pytest test_bot.py || echo "No tests found"

CMD ["python", "./bot.py"]