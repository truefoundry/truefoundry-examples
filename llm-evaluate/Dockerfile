FROM public.ecr.aws/truefoundrycloud/vllm:0.2.2-1.rc2
COPY requirements.txt /app/
RUN pip install -r /app/requirements.txt
WORKDIR /app/
COPY . /app/
