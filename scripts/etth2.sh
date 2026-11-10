#!/usr/bin/env bash
# Resolve data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

mkdir -p "logs/ETTh2"

model_name=CySAF

root_path_name=./dataset/ETT
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
seq_len=96

# 96 -> 96
seg_window=8
seg_stride=8
for window_type in 'rect'
do
for pred_len in 96
do
for random_seed in 2026 
do
    python -u run.py \
      --checkpoints "./checkpoints" \
      --results "./results/ETTh2" \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --train_epochs 15 \
      --patience 5 \
      --itr 1 \
      --batch_size 256 \
      --d_model 256 \
      --learning_rate 0.002 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --window_type $window_type \
      --random_seed $random_seed >logs/ETTh2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride'_'$window_type.log 

done
done
done 

# 96 -> 192
seg_window=8
seg_stride=8
for window_type in 'rect'
do
for pred_len in 192
do
for random_seed in 2026 
do
    python -u run.py \
      --checkpoints "./checkpoints/" \
      --results "./results/ETTh2" \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 7 \
      --cycle 24 \
      --train_epochs 15 \
      --patience 5 \
      --itr 1 \
      --batch_size 256 \
      --d_model 128 \
      --learning_rate 0.005 \
      --seg_window $seg_window \
      --seg_stride $seg_stride \
      --window_type $window_type \
      --random_seed $random_seed >logs/ETTh2/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len'_'$random_seed'_'$seg_window'_'$seg_stride'_'$window_type.log 

done
done
done 

