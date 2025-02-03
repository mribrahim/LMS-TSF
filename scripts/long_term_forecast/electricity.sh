

. ./scripts/long_term_forecast/define_params.sh
d_model=512

# Loop through each prediction length
for pred_len in "${pred_lengths[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/electricity/ \
    --data_path electricity.csv \
    --model_id ECL_$seq_len \
    --model $model_name \
    --channel_independence 0 \
    --d_model $d_model \
    --data custom \
    --features M \
    --seq_len $seq_len  \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 321 \
    --c_out 321 \
    --des 'Exp' \
    --train_epochs 20 \
    --itr 1
done
