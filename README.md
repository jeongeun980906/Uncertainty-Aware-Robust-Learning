# Elucidating Robust Learning with Uncertainty-Aware Corruption Pattern Estimation![image](https://user-images.githubusercontent.com/57895587/138092197-a404114c-0fcc-49e0-96ab-bda191e8ca00.png)

**Our contributions are as follows**

✔️ We propose a simple yet effective robust learning method leveraging a mixture of experts model on various noise settings.

✔️ The proposed method can not only robustly train from noisy data, but can also provide the explainability by discovering the underlying instance wise noise pattern within the dataset as well the two types of predictive uncertainties(aleatoric and epistemic)

✔️ We present set-dependent label noise setting, by applying label noise to only ambiguous set, by insight into annotators being confused at ambiguous inputs.

<p align="center">
  <img width="600" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/fig1.png">
</p>



## Introduction

📋 Official implementation of Explainable Robust Learning MLN

### 💡 Class Conditional Noise

**CIFAR100**
| Flipping Rate      | F-correction    | Co-teaching    | Co-teaching+   |    JoCoR       |    MLN(ours)   |
| ------------------ |---------------- | -------------- |----------------| -------------- | -------------- |
| Symmetry-20%       |   37.95±0.10    |   43.73±0.16   |  49.27±0.03    | **53.01±0.04** |   49.02±0.12   |
| Symmetry-50%       |   24.98±1.82    |   34.96±0.50   |  40.04±0.70    | **43.49±0.46** |   40.56±0.20   |
| Symmetry-80%       |   2.10±2.23     |   15.15±0.46   |  13.44±0.37    |   15.49±0.98   | **22.41±0.10** |
| Asymmetry-40%      |   25.94±0.44    |   28.35±0.25   |  33.62±0.39    |   32.79±0.35   | **34.51±0.10** |


**Noise Transition Matrix on CIFAR10**

asymmetric noise ; 
<p align="center">
  <img width="500" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/fig9-2.png">
</p>

### 💡 Set Dependent Noise

aleatoric uncertainty for the ambiguous set is higher than the clean set and larger for more label noise rate.

<p align="center">
  <img width="400" height="auto" src="https://github.com/jeongeun980906/Explainable-Robust-Learning-MLN/blob/master/misc/alea.png">
</p>

## Reproducing results of the paper

Example:

Asymmetry-40% on CIFAR10 

```
python3 main.py --data 'cifar10' --mode 'asymmetric' --ER 0.4

```


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
