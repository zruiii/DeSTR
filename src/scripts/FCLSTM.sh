###
 # @Author: zharui
 # @Date: 2023-09-05 15:06:50
 # @LastEditTime: 2023-09-06 10:10:07
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/FCLSTM.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
CUDA_VISIBLE_DEVICES=1 \
python main_forecast.py \
--city "zhuzhou" \
--forecast_model FCLSTM \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "guangzhou" \
--forecast_model FCLSTM \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "yizhuang" \
--forecast_model FCLSTM \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0


