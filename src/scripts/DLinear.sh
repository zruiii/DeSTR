CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "zhuzhou" \
--forecast_model DLinear \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--min_lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0



CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "guangzhou" \
--forecast_model DLinear \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--min_lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0



CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "yizhuang" \
--forecast_model DLinear \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--min_lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0


