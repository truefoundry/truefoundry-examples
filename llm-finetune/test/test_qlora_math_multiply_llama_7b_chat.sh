#!/bin/bash
set -ex
cd ..
python  train.py --model_id $MODEL_ID --output_dir $OUTPUT_DIR --report_to_mlfoundry true --ml_repo $ML_REPO --train_data $TRAIN_DATA --eval_size 0.1 --max_num_samples 0 --num_train_epochs 6 --per_device_train_batch_size 12 --per_device_eval_batch_size 12 --learning_rate 0.00005 --warmup_ratio 0.3 --gradient_accumulation_steps 1 --logging_steps 0.1 --logging_strategy steps --seed 42 --data_seed 42 --lr_scheduler_type linear --weight_decay 0.01 --max_grad_norm 1.0 --gradient_checkpointing true --save_strategy epoch --save_steps 500 --evaluation_strategy epoch --eval_steps 0.1 --log_checkpoints_to_mlfoundry true --lora_target_modules auto --half_precision_backend cuda_amp --cleanup_output_dir_on_start true  --use_qlora true --remove_unused_columns false  --tfy_run_name $RUN_NAME
# non finetuned
python test/test_evaluate_math_multiply.py --model_id $MODEL_ID --ml_repo $ML_REPO  --eval_data $EVAL_DATA --tfy_run_name $RUN_NAME --fine_tune false
# finetuned (model_id from disk - output_dir in train.py)
python test/test_evaluate_math_multiply.py --model_id $OUTPUT_DIR --ml_repo $ML_REPO  --eval_data $EVAL_DATA --tfy_run_name $RUN_NAME --fine_tune true
