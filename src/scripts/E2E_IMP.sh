###
 # @Author: zharui
 # @Date: 2023-09-01 19:46:34
 # @LastEditTime: 2023-09-02 00:32:56
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/E2E_IMP.sh
 # @Description: 
###


CUDA_VISIBLE_DEVICES=$1 \

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "zhuzhou" \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 50 \
--warmup_epochs 5 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 0.5 \
--obs_rec_weight 0.5


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "guangzhou" \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 50 \
--warmup_epochs 5 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 0.5 \
--obs_rec_weight 0.5


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "yizhuang" \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 32 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 50 \
--warmup_epochs 5 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 0.5 \
--obs_rec_weight 0.5











