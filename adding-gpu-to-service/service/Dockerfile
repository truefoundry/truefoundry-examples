FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
WORKDIR /gpu_test
COPY ./requirements.txt /tmp/
RUN pip install -U pip && pip install -U -r /tmp/requirements.txt
COPY . /gpu_test
CMD uvicorn main:app --host 0.0.0.0 --port 8000
