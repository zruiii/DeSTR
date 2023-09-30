###
 # @Author: zharui
 # @Date: 2023-08-15 08:51:37
 # @LastEditTime: 2023-09-12 21:30:34
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/PRE_IMP.sh
 # @Description: 
### 

CUDA_VISIBLE_DEVICES=$1 \

# DeSTR
# MODEL="ForecastModel"
# MODEL="../save/pretrain/pretrain_STMAE_20230825_1413.pth"

# PatchTST
MODEL_PATH="../save/pretrain/pretrain_PatchTST_20230911_1914.pth"
MODEL="PatchTST"

CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "zhuzhou" \
--load_from_pretrain \
--pretrain_model $MODEL \
--impute_model $MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 0.0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "guangzhou" \
--load_from_pretrain \
--pretrain_model $MODEL \
--impute_model $MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 128 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 0.0


CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
python main_impute.py \
--city "yizhuang" \
--load_from_pretrain \
--pretrain_model $MODEL \
--impute_model $MODEL \
--pretrain_path $MODEL_PATH \
--look_back 24 \
--pred_len 0 \
--time_dim 4 \
--batch_size 32 \
--lr 5e-4 \
--min_lr 5e-5 \
--max_epochs 100 \
--warmup_epochs 10 \
--eval_mask 30 50 70 \
--train_mask 0.2 \
--mask_imp_weight 1.0 \
--obs_rec_weight 0.0







