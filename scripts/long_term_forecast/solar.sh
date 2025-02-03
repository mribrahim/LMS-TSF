

. ./scripts/long_term_forecast/define_params.sh

# Loop through each prediction length
for pred_len in "${pred_lengths[@]}"; do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/Solar/ \
    --data_path solar_AL.txt \
    --model_id Solar_$seq_len \
    --model $model_name \
    --channel_independence 0 \
    --d_model $d_model \
    --data Solar \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 137 \
    --c_out 137 \
    --des 'Exp' \
    --itr 1 \
    --train_epochs 10
done
