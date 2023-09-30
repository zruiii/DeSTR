###
 # @Author: zharui
 # @Date: 2023-08-14 13:25:53
 # @LastEditTime: 2023-09-19 20:00:17
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/E2E_FOR.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "zhuzhou" \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "guangzhou" \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10 \


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "yizhuang" \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 32 \
--lr 1e-3 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10 \