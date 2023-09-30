###
 # @Author: zharui
 # @Date: 2023-08-21 16:08:05
 # @LastEditTime: 2023-09-02 00:35:47
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/GRIN.sh
 # @Description: 
###

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "zhuzhou" \
--impute_model GRIN \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 64 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.0 \
--mask_imp_weight 0 \
--obs_rec_weight 1


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "guangzhou" \
--impute_model GRIN \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 64 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.0 \
--mask_imp_weight 0 \
--obs_rec_weight 1


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "yizhuang" \
--impute_model GRIN \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 24 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.0 \
--mask_imp_weight 0 \
--obs_rec_weight 1




