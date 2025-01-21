#export CUDA_VISIBLE_DEVICES=0
model_name=iTransformer

pred_len=24

python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS03.npz \
  --model_id PEMS03 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in 358 \
  --dec_in 358 \
  --c_out 358 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --itr 1


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS04.npz \
  --model_id PEMS04 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 4 \
  --enc_in 307 \
  --dec_in 307 \
  --c_out 307 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.0005 \
  --itr 1 \


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS07.npz \
  --model_id PEMS07 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 883 \
  --dec_in 883 \
  --c_out 883 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --learning_rate 0.001 \
  --itr 1 \


python -u run.py \
 --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len 96 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 128 \
  --itr 1 \

