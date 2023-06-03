#### Test command

Run with Deepspeed: Pick from one of the deepspeed configs

```shell
deepspeed train.py \
          --output_dir ./model \
          --cleanup_output_dir_on_start \
          --max_num_samples 10 \
          --num_gpus 4 \
          --deepspeed ./1_ds_z2_config.json \
          --half_precision_backend cuda_amp \
          --model_id EleutherAI/pythia-70m \
          --report_to_mlfoundry false \
          --ml_repo transformers \
          --train_data https://assets.production.truefoundry.com/70k_samples.jsonl \
          --eval_data NA \
          --eval_size 0.1 \
          --num_train_epochs 3 \
          --per_device_train_batch_size 4 \
          --per_device_eval_batch_size 4 \
          --learning_rate 0.00005 \
          --warmup_ratio 0.3 \
          --gradient_accumulation_steps 1 \
          --logging_steps 0.1 \
          --logging_strategy steps\
          --seed 42 \
          --data_seed 42 \
          --lr_scheduler_type linear \
          --weight_decay 0.01 \
          --max_grad_norm 1.0 \
          --gradient_checkpointing true
```



#### For CPU

`--no_cuda` is broken on 4.29.2 :(

```shell
CUDA_VISIBLE_DEVICES=none python train.py \
                          --output_dir ./model \
                          --cleanup_output_dir_on_start \
                          --max_num_samples 10 \
                          --no_cuda \
                          --model_id EleutherAI/pythia-70m \
                          --report_to_mlfoundry false \
                          --ml_repo transformers \
                          --train_data https://assets.production.truefoundry.com/70k_samples.jsonl \
                          --eval_data NA \
                          --eval_size 0.1 \
                          --num_train_epochs 3 \
                          --per_device_train_batch_size 4 \
                          --per_device_eval_batch_size 4 \
                          --learning_rate 0.00005 \
                          --warmup_ratio 0.3 \
                          --gradient_accumulation_steps 1 \
                          --logging_steps 0.1 \
                          --logging_strategy steps\
                          --seed 42 \
                          --data_seed 42 \
                          --lr_scheduler_type linear \
                          --weight_decay 0.01 \
                          --max_grad_norm 1.0 \
                          --gradient_checkpointing true
```
