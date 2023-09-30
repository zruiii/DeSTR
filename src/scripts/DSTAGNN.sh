###
 # @Author: zharui
 # @Date: 2023-08-14 13:25:59
 # @LastEditTime: 2023-09-04 14:34:18
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/DSTAGNN.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "zhuzhou" \
--forecast_model DSTAGNN \
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
--forecast_model DSTAGNN \
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
--city "zhuzhou" \
--forecast_model DSTAGNN \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 32 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0
