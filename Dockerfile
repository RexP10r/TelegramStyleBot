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

RUN pip install --no-cache-dir \
    torch==2.7.1 \
    torchvision==0.22.1 \
    aiogram==3.20.0.post0 \
    pillow==11.2.1 \
    python-dotenv

WORKDIR /app
COPY . /app

ENV PYTHONUTF8=1
ENV PYTHONUNBUFFERED=1

CMD ["python", "./bot.py"]