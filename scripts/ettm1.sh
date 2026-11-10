#!/usr/bin/env bash
# Resolve data and output paths from the project root.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1


mkdir -p "logs/ETTm1"

model_name=CySAF
root_path_name=./dataset/ETT
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
seq_len=96
window_type=rect
random_seed=2026

# 96 -> 96
seg_window=8
seg_stride=8
python -u run.py \
      --checkpoints "./checkpoints" \
      --results "./results/ETTm1" \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_96' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 96 \
  --enc_in 7 \
  --cycle 96 \
  --train_epochs 15 \
  --patience 5 \
  --itr 1 \
  --batch_size 512 \
  --d_model 256 \
  --learning_rate 0.005 \
  --seg_window $seg_window \
  --seg_stride $seg_stride \
  --window_type $window_type \
  --mask_temperature 8 \
  --keep_bins 2 \
  --random_seed $random_seed \
  > logs/ETTm1/${model_name}_${model_id_name}_${seq_len}_96_${random_seed}_${seg_window}_${seg_stride}.log

# 96 -> 192
seg_window=6
seg_stride=6
python -u run.py \
      --checkpoints "./checkpoints" \
      --results "./results/ETTm1" \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_192' \
  --model $model_name \
  --data $data_name \
  --features M \
  --seq_len $seq_len \
  --pred_len 192 \
  --enc_in 7 \
  --cycle 96 \
  --train_epochs 30 \
  --patience 10 \
  --itr 1 \
  --batch_size 1024 \
  --d_model 512 \
  --learning_rate 0.0075 \
  --seg_window $seg_window \
  --seg_stride $seg_stride \
  --window_type $window_type \
  --mask_temperature 16 \
  --keep_bins 1 \
  --random_seed 2024 \
  > logs/ETTm1/${model_name}_${model_id_name}_${seq_len}_192_2024_${seg_window}_${seg_stride}.log
