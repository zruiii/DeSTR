###
 # @Author: zharui
 # @Date: 2023-09-11 15:31:33
 # @LastEditTime: 2023-09-11 19:13:27
 # @LastEditors: zharui@baidu.com
 # @FilePath: /decoupledST/src/scripts/PatchTST.sh
 # @Description: 
### 



CUDA_DEVICES=$1

#  pre-train
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
python main_pretrain.py \
--city yizhuang guangzhou zhuzhou \
--model PatchTST \
--look_back 24 \
--spatial_mask 0.2 \
--temporal_mask 0.2 \
--batch_size 48 \
--graph DIST \
--lr 5e-4 \
--min_lr 1e-5 \
--warmup_epochs 20 \
--max_epochs 200 \
--weight_decay 0.001 \
--use_ddp

# fine-tune
CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} \
python main_pretrain.py \
--city zhuzhou \
--model PatchTST \
--look_back 24 \
--spatial_mask 0.2 \
--temporal_mask 0.2 \
--batch_size 28 \
--graph DIST \
--lr 5e-4 \
--min_lr 1e-5 \
--warmup_epochs 20 \
--max_epochs 200 \
--weight_decay 0.001 \

