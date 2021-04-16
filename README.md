# [Reproducibility Challenge 2020](https://paperswithcode.com/rc2020): Replication of [ELECTRA](https://github.com/google-research/electra)

This repository contains a reimplementation in PyTorch of [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) for the [Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).
This project was undertaken as part of the course [IFT6268 Self Supervised Representation Learning](https://sites.google.com/view/ift6268-a2020/schedule?authuser=0) at [Mila / University of Montreal](https://mila.quebec).

The preprocessing process is embedding and cached with the command lines for pretraining and downstream tasks.

A pretrained model with 1M steps training is also available via this [link](https://wandb.ai/cccwam/rc2020_electra_pretraining/artifacts/model/pretrained_model/0657b16512c9728d08c0/files).

This work leverages HuggingFace libraries (Transformers, Datasets, Tokenizers) and PyTorch (1.7.1).

For more information, please refer to the associated [paper](https://arxiv.org/abs/2104.02756).

## Main results

My results are similar to the original ELECTRA’s implementation (Clark et al. [2020]), despite minor differences compared to the original paper for both implementations. With only 14M parameters, ELECTRA outperforms, in absolute performances, concurrent pretraining approaches from some previous SOTA, such as GPT, or alternative efficient approaches using knowledge distillation, such as DistilBERT. By taking into account compute cost, ELECTRA is clearly outperforming all compared approaches, including BERT and TinyBERT. Therefore, this work supports the claim that ELECTRA achieves high level of performances in low-resource settings, in term of compute cost. Furthermore, with an increased generator capacity than recommended by Clark et al. [2020], the discriminant can collapses by being unable to distinguish if inputs are fake or not. Thus, while ELECTRA is easier to train than GAN (Goodfellow et al. [2014]), it appears to be sensitive to capacity allocation between generator and discriminator.


### Training behaviour

 Original implementation
![Training behaviour - Original](https://user-images.githubusercontent.com/1091306/76335698-256fb500-62b2-11ea-9fee-e39aca5cae24.png)

This implementation
![Training behaviour - Mine](https://github.com/cccwam/rc2020_electra/blob/latest_branch/images/Electra%20RC2020%20-%20Learning.png)

More details are available in [WandB](https://wandb.ai/cccwam/rc2020_electra_pretraining/reports/RC2020-Replication-of-ELECTRA-Clark-et-al-2020---VmlldzozODYzMjk)

### Results on GLUE dev set

| Model | CoLA (Mcc)  | SST-2 (Acc)   | MRPC (Acc)  | STS-B(Spc)   | QQP (Acc)  | MNLI (Acc)  | QNLI (Acc)  | RTE (Acc)   | AVG | GLUE |   
|-------|------|-------|------|-------|------|------|------|------ |-----|-------|
|Original| 56.8 | 88.3 | 87.4 | 86.8 | 88.3 | 78.9 | 87.9 | 68.5 | 80.4 | |
|Mine    | 53.5 | 88.7 | 87.6 | 85.2 | 86.1 | 80.2 | 87.5 | 61.5 | 79.2 | 76.7 |   

### Efficiency analysis

| Model | # Params | Training time + hardware | pfs-days | AVG  | GLUE | pfs-days per AVG | pfs-days per GLUE| 
|-------|------|-------|------|-------|------|------|------|
|GPT| 110M | 30d on 8 P600     | 0.95 | 77.9 | 75.4   | 0.17 | 0.18  |
|DistilBERT| 67M | 90h on 8 V100     | 0.16 |  | 77.0  |  | 0.21 |
|ELECTRA-Original| 14M | 4d on 1 V100     | 0.02 | 80.4 |      | 0.03 |  |
|ELECTRA-Mine    | 14M | 3.75d on 1 RTX 3090 | 0.03 | 79.2 | 76.7 | 0.05 | 0.06 |   


## Main differences with Electra original implementation

- Preprocessing. This reimplementation caches the tokenization step and dynamically pick a random segment during training. The segmentation is therefore dynamic instead of static in the original implementation. Furthermore, this reimplementation handles completely the download of pretraining datasets with [HuggingFace datasets library](https://github.com/huggingface/datasets).
- Fine-tuning, the original implementation has got a discrepancy with the paper for the layerwise learning rate decay, see [Github](https://github.com/google-research/electra/issues/51).
- Task specific data augmentation. The original implementation uses a technique, called double_unordered, to increase by 2 the dataset for MRPC and STS. This implementation doesn't use any task specific data augmentation, see [Github](https://github.com/google-research/electra/issues/98).


For more information, please refer to the [paper (under review)](To be added later).


## How to cite this work

```
@misc{
mercier2021efficient,
title={Efficient transfer learning for {NLP} with {ELECTRA}},
author={Fran{\c{c}}ois MERCIER},
year={2021},
url={https://openreview.net/forum?id=Or5sv1Pj6od}
}
```

## Experiment 1: Reproduce ElectraSmall

In this experiment, we use the maximum sequence length of 128 like ElectraSmall and the masking strategy is 15% of input tokens with 85% chance of replacement.

### Pretraining

```
python run_pretraining.py --mlm_probability 0.15  --mlm_replacement_probability 0.85 --max_length 128 --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --logging_steps 3840 --eval_steps 12800 --save_steps 1280000 --experiment_name pretraining_OWT --generator_layer_size 1.0 --generator_size 0.25
```

### Downstream tasks: Glue 

```
python run_glue.py  --experiment_name "electra_replication_6-25" --pretrain_path pretrained_model/checkpoint-1000000
```

## Experiment 2: Sensitivity to generator

#### Pretraining for 12.5% generator size

```
python run_pretraining.py --mlm_probability 0.15  --mlm_replacement_probability 0.85 --max_length 128 --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --logging_steps 3840 --eval_steps 12800 --save_steps 1280000 --experiment_name pretraining_OWT --generator_layer_size 1.0 --generator_size 0.125
```

#### Pretraining for 50% generator size

```
python run_pretraining.py --mlm_probability 0.15  --mlm_replacement_probability 0.85 --max_length 128 --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --logging_steps 3840 --eval_steps 12800 --save_steps 1280000 --experiment_name pretraining_OWT --generator_layer_size 1.0 --generator_size 0.5
```

#### Pretraining for 75% generator size

```
python run_pretraining.py --mlm_probability 0.15  --mlm_replacement_probability 0.85 --max_length 128 --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --logging_steps 3840 --eval_steps 12800 --save_steps 1280000 --experiment_name pretraining_OWT --generator_layer_size 1.0 --generator_size 0.75
```

#### Pretraining for 100% generator size

```
python run_pretraining.py --mlm_probability 0.15  --mlm_replacement_probability 0.85 --max_length 128 --per_device_train_batch_size 128 --gradient_accumulation_steps 1 --logging_steps 3840 --eval_steps 12800 --save_steps 1280000 --experiment_name pretraining_OWT --generator_layer_size 1.0 --generator_size 1.0
```

## How to pretrain tokenizer - Optional since one is already provided in this repository

`python train_tokenizer.py --output_dir models`


## Requirements

```
conda install pytorch=1.7.1 torchvision torchaudio -c pytorch
pip install -r requirements
```

## Additional resources

- [ML Reproducibility Challenge 2020](https://paperswithcode.com/rc2020)

- [Original implementation](https://github.com/google-research/electra)
- [Reimplementation of ELECTRA with FastAI and PyTorch](https://github.com/richarddwang/electra_pytorch)
