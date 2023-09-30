###
 # @Author: zharui
 # @Date: 2023-06-15 09:51:19
 # @LastEditTime: 2023-09-06 19:18:53
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/run_pretrain.sh
 # @Description: 
###



CUDA_DEVICES=$1

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
python main_pretrain.py \
--city yizhuang guangzhou zhuzhou \
--look_back 24 \
--mask_mode "spacetime" \
--mask_ratio 0.4 \
--batch_size 28 \
--graph DIST \
--lr 1e-4 \
--min_lr 1e-5 \
--warmup_epochs 20 \
--max_epochs 200 \
--weight_decay 0.001 \
--use_ddp

