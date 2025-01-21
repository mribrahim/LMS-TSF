export CUDA_VISIBLE_DEVICES=0

batch_size=128
down_sampling_layers=3
down_sampling_window=2

for method in CrossScaleNet #LMSAutoTSF iTransformer TimeMixer PatchTST
do
  python -u run.py \
    --task_name anomaly_detection \
    --is_training 1 \
    --root_path ./dataset/PSM \
    --model_id PSM \
    --model $method \
    --data PSM \
    --features M \
    --seq_len 100 \
    --pred_len 0 \
    --d_model 32 \
    --d_ff 32 \
    --e_layers 3 \
    --enc_in 25 \
    --c_out 25 \
    --down_sampling_layers $down_sampling_layers \
    --down_sampling_method avg \
    --down_sampling_window $down_sampling_window \
    --batch_size $batch_size \
    --anomaly_ratio 1 \
    --train_epochs 20
  done