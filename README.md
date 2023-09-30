# Scaling up Multivariate Time Series Pre-Training with Decoupled Spatial-Temporal Representations

This is a PyTorch/GPU re-implementation of our paper: "Scaling up Multivariate Time Series Pre-Training with Decoupled Spatial-Temporal Representations"

![image-20230930175638783](https://raw.githubusercontent.com/zruiii/PicCloud/main/Typora-img/202309301756754.png)



### Training from scratch

```shell
cd src
sh script/E2E_FOR gpu
sh script/E2E_IMP gpu
```

Please set the parameters in the scripts, such as the target city.

The reproductivity of all baselines are also provided in `baselines` folder, you can run the baseline models using corresponding script.



### Pre-training DeSTR

```shell
cd src
sh run_pretrain.sh gpus
```

**Distributed Data Parallel Training** is supported for the pre-training, where you can replace `gpus` with the GPU devices, such as `0,1,2,3` utilizes 4 GPUs for training.

More parameters can be set in the script `run_pretrain.sh`, such as the pre-training crops (zhuzhou/guangzhou/yizhuang), sampling strategy (time-only/space-only/spacetime-augnostic), etc.

We provide a pre-trained model in the fold `save/`



### Fine-tuning DeSTR

```shell
cd src

sh script/PRE_FOR gpu
sh script/PRE_IMP gpu
```

You can use the pre-trained model we provided to perform the fine-tuning.

We also provide two fine-tuned model in the fold `save/` to achieve 12-step forecasting and 50% imputation.




