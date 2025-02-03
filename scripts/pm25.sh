learning_rate=0.003
d_model=32
batch_size=32
train_epochs=20

seq_len=24
pred_lengths=(12 24 48)


model_list=(LMSAutoTSFV2)

for model_name in "${model_list[@]}"; do
    for pred_len in "${pred_lengths[@]}"; do
        python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/ \
        --model $model_name \
        --data PM25 \
        --data_path Beijing-PM2.5.csv \
        --model_id PM25_$pred_len \
        --seq_len $seq_len \
        --label_len 0 \
        --pred_len $pred_len \
        --enc_in 11 \
        --c_out 1 \
        --features MS \
        --channel_independence 0 \
        --des 'Exp' \
        --itr 1 \
        --learning_rate $learning_rate \
        --d_model $d_model \
        --batch_size $batch_size \
        --train_epochs $train_epochs \
        --output_attention
        # --explainer None
    done
done