#export CUDA_VISIBLE_DEVICES=0

model_name=LMSAutoTSF

seq_len=96
pred_len=48

down_sampling_layers=3
down_sampling_window=2
learning_rate=0.003
d_model=128
batch_size=32
train_epochs=20
patience=3

python -u run.py \
 --task_name long_term_forecast \
 --is_training 0 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS03.npz \
 --model_id PEMS03 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --enc_in 358 \
 --dec_in 358 \
 --c_out 358 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
  --channel_independence 0 \
 --d_model $d_model \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u run.py \
 --task_name long_term_forecast \
 --is_training 0 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS04.npz \
 --model_id PEMS04 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --enc_in 307 \
 --dec_in 307 \
 --c_out 307 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model $d_model \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u run.py \
 --task_name long_term_forecast \
 --is_training 0 \
 --root_path ./dataset/PEMS/ \
 --data_path PEMS07.npz \
 --model_id PEMS07 \
 --model $model_name \
 --data PEMS \
 --features M \
 --seq_len $seq_len \
 --label_len 0 \
 --pred_len $pred_len \
 --enc_in 883 \
 --dec_in 883 \
 --c_out 883 \
 --des 'Exp' \
 --itr 1 \
 --use_norm 0 \
 --channel_independence 0 \
 --d_model $d_model \
 --batch_size 32 \
 --learning_rate $learning_rate \
 --train_epochs $train_epochs \
 --patience $patience \
 --down_sampling_layers $down_sampling_layers \
 --down_sampling_method avg \
 --down_sampling_window $down_sampling_window


python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/PEMS/ \
  --data_path PEMS08.npz \
  --model_id PEMS08 \
  --model $model_name \
  --data PEMS \
  --features M \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --enc_in 170 \
  --dec_in 170 \
  --c_out 170 \
  --des 'Exp' \
  --itr 1 \
  --use_norm 0 \
  --channel_independence 0 \
  --d_model $d_model \
  --batch_size 32 \
  --learning_rate $learning_rate \
  --train_epochs 10 \
  --patience $patience \
  --down_sampling_layers $down_sampling_layers \
  --down_sampling_method avg \
  --down_sampling_window $down_sampling_window
