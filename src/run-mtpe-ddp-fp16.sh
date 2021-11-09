#!/bin/bash
exp_setting=BARTbase-fp16-baseline
exp_setting=BARTbase-fp16-debug
lr=5e-6
exp_setting=MBARTlarge-fp16-debug-$lr
DATAPATH=../data
OUTPUT=outputs/run-MTPE-$exp_setting
mkdir -p $OUTPUT
CUDA_VISIBLE_DEVICES=5,6 python -u -m torch.distributed.launch --master_port 88888 --nproc_per_node 2 run-mtpe.py \
    --model_name_or_path $1 \
    --do_train \
    --do_eval \
    --train_file $DATAPATH/train.jsonl \
    --validation_file $DATAPATH/dev.jsonl \
    --test_file $DATAPATH/test.jsonl \
    --source_prefix "" \
    --output_dir $OUTPUT \
    --learning_rate $lr \
    --lr_scheduler polynomial \
    --ignore_pad_token_for_loss "True" \
    --warmup_steps 200 \
    --num_train_epochs 15 \
    --save_strategy "epoch" \
    --metric_for_best_model "eval_bleu" \
    --evaluation_strategy "epoch" \
    --save_total_limit 1 \
    --gradient_accumulation_steps 4 \
    --per_device_train_batch_size=2 \
    --per_device_eval_batch_size=2 \
    --weight_decay 0.01 \
    --label_smoothing_factor 0.1 \
    --max_grad_norm 0.1 \
    --dropout 0.1 \
    --num_beams 5 \
    --overwrite_output_dir \
    --max_source_length=256 \
    --max_target_length=100 \
    --val_max_target_length=100 \
    --logging_strategy "steps" \
    --logging_steps 10 \
    --fp16 \
    --load_best_model_at_end \
    --predict_with_generate 2>&1 | tee $OUTPUT/run.log
