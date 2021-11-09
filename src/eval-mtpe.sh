#!/bin/bash

SETTING=trainer-DDP-fp16
DATAPATH=../data
OUTPUT=outputs/eval-MTPE-t5-$SETTING
mkdir -p $OUTPUT
echo "save in $OUTPUT ..."
CUDA_VISIBLE_DEVICES=1,4 python -u run-mtpe-t5.py \
	--model_name_or_path $1 \
	--do_eval \
    --do_predict \
	--train_file $DATAPATH/train.jsonl \
    --validation_file $DATAPATH/dev.jsonl \
    --test_file $DATAPATH/test.jsonl \
    --source_prefix "" \
	--output_dir $OUTPUT \
    --learning_rate 3e-5 \
    --lr_scheduler polynomial \
    --warmup_steps 200 \
    --num_train_epochs 15 \
    --save_strategy epoch \
    --metric_for_best_model "bleu" \
    --gradient_accumulation_steps 8 \
	--per_device_train_batch_size=8 \
	--per_device_eval_batch_size=4 \
    --weight_decay 0.01 \
    --max_grad_norm 0.1 \
    --label_smoothing_factor 0.1 \
    --num_beams 5 \
	--overwrite_output_dir \
    --max_source_length=256 \
    --max_target_length=100 \
    --val_max_target_length=100 \
	--predict_with_generate 2>&1 | tee $OUTPUT/eval.log
