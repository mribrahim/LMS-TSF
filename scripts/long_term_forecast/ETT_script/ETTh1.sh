

. ./scripts/long_term_forecast/define_params.sh



# Loop through each prediction length
for pred_len in "${pred_lengths[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_$seq_len \
    --model $model_name \
     --d_model $d_model \
    --channel_independence 0 \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --train_epochs 20 \
    --itr 1
  done
