FROM --platform=linux/amd64 mambaorg/micromamba:1.5.3-jammy
USER root
RUN apt update && apt install -y git
USER mambauser
RUN micromamba install -y -c "nvidia/label/cuda-11.7.0" cuda-nvcc -n base && \
    micromamba install -y -c "conda-forge" python=3.10 -n base
ARG MAMBA_DOCKERFILE_ACTIVATE=1
COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip wheel setuptools && \
    pip install --no-cache-dir -U -r /tmp/requirements.txt && \
    pip install --no-cache-dir --no-build-isolation -U flash-attn==2.3.4
# This is a hacky work around! Ideal way is to use
# /usr/local/bin/_entrypoint.sh as the main entrypoint (command in K8s)
ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"
WORKDIR /app
COPY . /app
