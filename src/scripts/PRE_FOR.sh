###
 # @Author: zharui
 # @Date: 2023-08-15 08:51:37
 # @LastEditTime: 2023-09-19 09:24:28
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/PRE_FOR.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

# DeSTR
PRE_MODEL="DeSTR"
FOR_MODEL="ForecastModel"
MODEL_PATH="../save/DeSTR_pretrain.pth"

# PatchTST
# MODEL_PATH="../save/pretrain/pretrain_PatchTST_20230911_1914.pth"
# FOR_MODEL="PatchTST"
# PRE_MODEL="PatchTST"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_forecast.py \
--city "zhuzhou" \
--load_from_pretrain \
--pretrain_model $PRE_MODEL \
--forecast_model $FOR_MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10


CUDA_VISIBLE_DEVICES=2 \
python main_forecast.py \
--city "guangzhou" \
--load_from_pretrain \
--pretrain_model $MODEL \
--forecast_model $MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10


CUDA_VISIBLE_DEVICES=7 \
python main_forecast.py \
--city "yizhuang" \
--load_from_pretrain \
--pretrain_model $MODEL \
--forecast_model $MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 12 \
--time_dim 4 \
--batch_size 48 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10

