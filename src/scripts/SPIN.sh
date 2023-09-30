###
 # @Author: zharui
 # @Date: 2023-09-03 14:11:59
 # @LastEditTime: 2023-09-03 14:18:29
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/SPIN.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1

CUDA_VISIBLE_DEVICES=2 \
python main_impute.py \
--city "zhuzhou" \
--impute_model SPIN \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 12 \
--time_encode cyclic \
--time_dim 8 \
--lr 5e-4 \
--min_lr 5e-4 \
--max_epochs 100 \
--warmup_epochs 0 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 0.0

