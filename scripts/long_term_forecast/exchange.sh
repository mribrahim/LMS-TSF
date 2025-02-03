

. ./scripts/long_term_forecast/define_params.sh


# Loop through each prediction length
for pred_len in "${pred_lengths[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --model_id Exchange_$seq_len \
    --model $model_name \
    --channel_independence 1 \
    --d_model $d_model \
    --data custom \
    --seq_len $seq_len  \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 8 \
    --c_out 8 \
    --des 'Exp' \
    --itr 1
done
