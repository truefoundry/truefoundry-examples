FROM --platform=linux/amd64 pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -U pip setuptools wheel && \
    pip install -r /app/requirements.txt
COPY . /app
