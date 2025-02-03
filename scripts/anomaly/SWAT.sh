export CUDA_VISIBLE_DEVICES=0

batch_size=128
down_sampling_layers=3
down_sampling_window=2

# for method in CrossScaleNet LMSAutoTSF iTransformer TimeMixer PatchTST
for method in LMSAutoTSF
do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 0 \
    --root_path ./dataset/SWAT \
    --model_id SWAT \
    --model $method \
    --data SWAT \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 32 \
    --d_ff 32 \
    --e_layers 3 \
    --enc_in 51 \
    --c_out 51 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --batch_size $batch_size \
    --anomaly_ratio 1 \
    --train_epochs 20
  done