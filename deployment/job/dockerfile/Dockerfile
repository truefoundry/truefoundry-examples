FROM --platform=linux/amd64 python:3.9.14-slim
WORKDIR /job
COPY requirements.txt /tmp/
RUN pip install -U pip && pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /job/
CMD python /job/train.py

