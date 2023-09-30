###
 # @Author: zharui
 # @Date: 2023-08-23 16:14:46
 # @LastEditTime: 2023-09-06 14:41:29
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/SAITS.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=1 \
python main_impute.py \
--city "zhuzhou" \
--impute_model SAITS \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 1.0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "guangzhou" \
--impute_model SAITS \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 1.0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "yizhuang" \
--impute_model SAITS \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 1e-3 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 1.0

