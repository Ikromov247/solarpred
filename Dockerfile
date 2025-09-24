FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# all files will be copied here 
WORKDIR /app

# set timezone
ENV TZ=Asia/Seoul
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && \
    echo $TZ > /etc/timezone && \
    dpkg-reconfigure --frontend noninteractive tzdata

# install python and uv
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

COPY . /app

RUN pip3 install --no-cache-dir -r /app/requirements.txt

ENTRYPOINT [ "python3", "solar_pred/main.py"]