#### Test command

```shell
HF_HOME=/data/hfhome/ deepspeed \
                      --num_gpus 4 \
                      train.py \
                      --deepspeed ./ds_z2_config.json \
                      --half_precision_backend cuda_amp \
                      --gradient_checkpointing \
                      --model_id EleutherAI/pythia-70m \
                      --train_data https://assets.production.truefoundry.com/70k_samples.jsonl \
                      --ml_repo abcd \
                      --per_device_train_batch_size 64 \
                      --num_train_epochs 1 \
                      --max_num_samples 200
```

- Pick from one of the deepspeed configs

#### For CPU

`--no_cuda` is broken on 4.29.2 :(

```shell
CUDA_VISIBLE_DEVICES=none HF_HOME=/data/hfhome/ python
                      train.py \
                      --no_cuda
                      --gradient_checkpointing \
                      --model_id EleutherAI/pythia-70m \
                      --train_data https://assets.production.truefoundry.com/70k_samples.jsonl \
                      --ml_repo abcd \
                      --per_device_train_batch_size 64 \
                      --num_train_epochs 1
                      --max_num_samples 200

```