FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# all files will be copied here 
WORKDIR /app

# set timezone
ENV TZ=Asia/Seoul
ENV PYTHONPATH=/app

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


COPY . /app

RUN pip3 install -r /app/requirements.txt
RUN pip3 install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1

# Set PYTHONPATH to include the app directory so solar_pred module can be found

ENTRYPOINT [ "python3", "-m", "solar_pred.main"]