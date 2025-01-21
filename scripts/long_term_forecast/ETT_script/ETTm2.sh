

. ./scripts/long_term_forecast/define_params.sh

# Loop through each prediction length
for pred_len in "${pred_lengths[@]}"; do
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --model_id ETTm2_$seq_len_$pred_len \
    --model $model_name \
    --channel_independence 0 \
    --d_model $d_model \
    --data ETTm2 \
    --features M \
    --seq_len $seq_len  \
    --label_len 96 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --c_out 7 \
    --des 'Exp' \
    --train_epochs 20 \
    --itr 1
done

