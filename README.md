# Elucidating Robust Learning with Uncertainty-Aware Corruption Pattern Estimation

**Our contributions are as follows**

✔️ We propose a simple yet effective robust learning method leveraging a mixture of experts model on various noise settings.

✔️ The proposed method can not only robustly train from noisy data, but can also provide the explainability by discovering the underlying instance wise noise pattern within the dataset as well the two types of predictive uncertainties(aleatoric and epistemic)

✔️ We present a novel evaluation scheme for validating the set-dependent corruption pattern estimation performance.

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/fig1.png">
</p>


## Introduction

📋 Official implementation of Explainable Robust Learning MLN

### Requirements

```
torch==1.7.1
torchvision==0.8.2
matplotlib==3.4.1
scikit-learn==0.24.1
gensim==4.0.1
scipy==1.6.2
seabotn==0.11.1
Pillow==8.2.0
```
### Datasets
Please download mannually TREC dataset
**TREC**
https://cogcomp.seas.upenn.edu/Data/QA/QC/

## Reproducing results of the paper

e.g., mnist on class conditional noise setting

```
cd scripts
./ccn_mnist.sh

```

### 💡 Class Conditional Noise

**CIFAR10**
| Flipping Rate      | F-correction    | Co-teaching    | Co-teaching+   |    JoCoR       |    MLN(ours)   |
| ------------------ |---------------- | -------------- |----------------| -------------- | -------------- |
| Symmetry-20%       |   68.74±0.20    |   78.23±0.27   |  78.71±0.34    | **85.73±0.19** |   84.20±0.05   |
| Symmetry-50%       |   42.71±0.42    |   71.30±0.13   |  57.05±0.54    | **79.41±0.25** |   77.88±0.07   |
| Symmetry-80%       |   15.88±0.42    |   26.58±2.22   |  24.19±2.74    |   27.78±3.06   | **41.83±0.10** |
| Asymmetry-40%      |   70.60±0.40    |   73.78±0.22   |  68.84±0.20    |   76.36±0.49   | **76.62±0.07** |


**Noise Transition Matrix on CIFAR10**

<p align="center">
  <img width="500" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/cifar10_tm.png">
</p>

### 💡 Set Dependent Noise

aleatoric uncertainty for the ambiguous set is higher than the clean set and larger for more label noise rate.
<p align="center">
  <img width="400" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/alea.png">
</p>

estimated noise transition matrix for partioned sets are:
<p align="center">
  <img width="400" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/dirty_mnist_tm.png">
</p>

<p align="center">
  <img width="400" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/dirty_cifar10_tm.png">
</p>

## Citing our paper

If you find this work useful please consider citing it:

```
@article{papername,
  title={title},
  author={authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2021}
}
```
