# [Reproducibility Challenge 2020](https://paperswithcode.com/rc2020): Replication of [ELECTRA](https://github.com/google-research/electra)

This repository contains a reimplementation in PyTorch of [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) for the [Reproducibility Challenge 2020](https://paperswithcode.com/rc2020).
This project was undertaken as part of the course [IFT6268 Self Supervised Representation Learning](https://sites.google.com/view/ift6268-a2020/schedule?authuser=0) at [Mila / University of Montreal](https://mila.quebec).

The preprocessing process is embedding and cached with the command lines for pretraining and downstream tasks.

This work leverages HuggingFace libraries (Transformers, Datasets, Tokenizers) and PyTorch (1.7.0).

For more information, please refer to the associated [paper (under review)](To be added later).

## Main results

My results are similar to the original ELECTRA’s implementation (Clark et al. [2020]), despite minor differences compared to the original paper for both implementations. With only 14M parameters, ELECTRA outperforms, in absolute performances, concurrent pretraining approaches from some previous SOTA, such as GPT, or alternative efficient approaches using knowledge distillation, such as DistilBERT. By taking into account compute cost, ELECTRA is clearly outperforming all compared approaches, including BERT and TinyBERT. Therefore, this work supports the claim that ELECTRA achieves high level of performances in low-resource settings, in term of compute cost. Furthermore, with an increased generator capacity than recommended by Clark et al. [2020], the discriminant can collapses by being unable to distinguish if inputs are fake or not. Thus, while ELECTRA is easier to train than GAN (Goodfellow et al. [2014]), it appears to be sensitive to capacity allocation between generator and discriminator.


### Training behaviour

 Original implementation
![Training behaviour - Original](https://github.com/cccwam/ift6268/blob/ReleaseCode/images/Electra%20RC2020%20-%20Learning%20-%20Original.png)

This implementation
![Training behaviour - Mine](https://github.com/cccwam/ift6268/blob/ReleaseCode/images/Electra%20RC2020%20-%20Learning.png)

### Results on GLUE dev set

| Model | CoLA (Mcc)  | SST-2 (Acc)   | MRPC (Acc)  | STS-B(Spc)   | QQP (Acc)  | MNLI (Acc)  | QNLI (Acc)  | RTE (Acc)   | AVG | GLUE* |   
|-------|------|-------|------|-------|------|------|------|------ |-----|-------|
|Original| 56.8 | 88.3 | 87.4 | 86.8 | 88.3 | 78.9 | 87.9 | 68.5 | 80.4 | |
|Mine    | 51.9 | 88.5 | 90.7 | 85.5 | 85.8 | 79.5 | 87.8 | 62.2 | 79.0 | 76.5 |   

### Efficiency analysis

| Model | # Params | Training time + hardware | pfs-days | AVG  | GLUE | pfs-days per AVG | pfs-days per GLUE| 
|-------|------|-------|------|-------|------|------|------|
|GPT| 110M | 30d on 8 P600     | 0.95 | 77.9 | 75.4   | 0.17 | 0.18  |
|DistilBERT| 67M | 90h on 8 V100     | 0.16 |  | 77.0  |  | 0.21 |
|ELECTRA-Original| 14M | 4d on 1 V100     | 0.02 | 80.4 |      | 0.03 |  |
|ELECTRA-Mine    | 14M | 5d on 1 RTX 3090 | 0.03 | 79.0 | 76.5 | 0.03 | 0.03 |   


## Main differences with Electra original implementation

- Preprocessing. This reimplementation uses Spacy for sentences segmentation whereas the original implementation don't perform sentence segmentations. 
- Use of relative position embeddings in addition to absolute position embeddings.
- Use of sentence embeddings to refer the id of the sentence (similar to token type id in Electra/Bert but with more ids)

For more information, please refer to the [paper (under review)](To be added later).


## How to cite this work

```
@misc{cccwamif93:online,
author = {François Mercier},
title = {ML Reproducibility Challenge 2020: Electra reimplementation using PyTorch and Transformers},
howpublished = {\url{https://github.com/cccwam/ift6268}},
month = {},
year = {2020},
note = {(Accessed on 12/07/2020)}
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
conda install pytorch=1.7.0 torchvision torchaudio -c pytorch
pip install -r requirements
```
