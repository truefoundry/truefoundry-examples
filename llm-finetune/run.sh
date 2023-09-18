#!/usr/bin/env bash
python train.py \
      --output_dir ./model \
      --cleanup_output_dir_on_start \
      --max_num_samples 100 \
      --no_cuda \
      --model_id EleutherAI/pythia-70m \
      --report_to_mlfoundry false \
      --ml_repo transformers \
      --train_data file:///Users/chiragjn/Downloads/standford_alpaca_train_49k.jsonl \
      --eval_data file:///Users/chiragjn/Downloads/standford_alpaca_test_2k.jsonl \
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
      --gradient_checkpointing true \
      --save_total_limit 3 \
      --use_lora true \
      --lora_config '{"r": 8, "lora_alpha": 32, "target_modules": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], "lora_dropout": 0.05, "bias": "none", "task_type": "CAUSAL_LM"}'
