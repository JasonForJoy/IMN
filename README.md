# Interactive Matching Network for Multi-Turn Response Selection
This repository contains the source code and datasets for the CIKM 2019 paper [Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots](https://arxiv.org/pdf/1901.01824.pdf) by Gu et al. <br>

Our proposed Interactive Matching Network (IMN) has achieved a new state-of-the-art performance on four large-scale datasets that are publicly available for research on multi-turn conversation.

## Model overview
<img src="image/model.png">

## Results
<img src="image/UbuntuV1_V2.png">
<img src="image/Douban_Ecommerce.png">

## Dependencies
Python 2.7 <br>
Tensorflow 1.4.0

## Datasets
Your can download the processed datasets used in our paper here and unzip it to the folder of ```data```. <br>
[Ubuntu_V1](https://drive.google.com/open?id=1-rNv34hLoZr300JF3v7nuLswM7GRqeNc) <br>
[Ubuntu_V2](https://drive.google.com/open?id=1tS_VC47z8CVPr-tZu0U4JEEwBT04N6ks) <br>
[Douban](https://drive.google.com/open?id=1Cwt5BC_WDr1N_-TYaOMSHuOXLKAxXoMQ) <br>
[Ecommerce](https://drive.google.com/open?id=1vy2bcTCLm1Dzsdvh0cvPIw0XzrTK06us)

## Train a new model
Take Ubuntu_V1 as an example.
```
cd scripts
bash ubuntu_train.sh
```
The training process is recorded in ```log_train_IMN_UbuntuV1.txt``` file.

## Test a trained model
```
bash ubuntu_test.sh
```
The testing process is recorded in ```log_test_IMN_UbuntuV1.txt``` file. And your can get a ```ubuntu_test_out.txt``` file which records scores for each context-response pair. Run the following command and you can compute the metric of Recall.
```
python compute_recall.py
```

## Cite
If you use the code and datasets, please cite the following paper:
**"Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots"**
Jia-Chen Gu, Zhen-Hua Ling, Quan Liu. _CIKM (2019)_

```
@inproceedings{gu2019interactive,
  title        = {Interactive Matching Network for Multi-Turn Response Selection in Retrieval-Based Chatbots},
  author       = {Jia{-}Chen Gu and
                  Zhen{-}Hua Ling and
                  Quan Liu},
  booktitle    = {Proceedings of the 28th ACM International Conference on Information and Knowledge Management},
  year         = {2019},
  organization = {ACM}
}
```
