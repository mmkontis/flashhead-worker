FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git ffmpeg libgl1 libglib2.0-0 libsndfile1 curl build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN python3.10 -m pip install --no-cache-dir -U pip
RUN python3.10 -m pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir -r /tmp/requirements.txt
RUN python3.10 -m pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310-cp310-linux_x86_64.whl \
 || python3.10 -m pip install --no-cache-dir \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN git clone --depth 1 https://github.com/Soul-AILab/SoulX-FlashHead.git /app/SoulX-FlashHead
COPY handler.py /app/handler.py
WORKDIR /app/SoulX-FlashHead
CMD ["python3.10", "-u", "/app/handler.py"]
