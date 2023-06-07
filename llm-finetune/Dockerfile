FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /app
RUN apt update && \
    apt install -y git && \
    rm -rf /var/lib/apt/lists/* && \
    pip install -U pip setuptools wheel
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
