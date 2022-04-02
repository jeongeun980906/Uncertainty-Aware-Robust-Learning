from dataloader.mnist import MNIST
from torchvision import datasets,transforms
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns

NOISE_RATE = 0.2
dataset = MNIST(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                        noise_type='instance',noise_rate=NOISE_RATE)

print(len(dataset.train_labels), len(dataset.train_noisy_labels))
clean_labels = np.asarray(dataset.train_labels)
noisy_labels = np.asarray(dataset.train_noisy_labels)

TM = np.zeros((10,10))
for i,j in zip(dataset.train_labels,dataset.train_noisy_labels):
    TM[i,j]+=1
temp = np.sum(TM, axis=-1)
TM /= temp
TM = np.round(TM,2)

plt.figure()
plt.title('IDN {} confusion matrix'.format(NOISE_RATE))
plt.xlabel("Noise Label")
plt.ylabel("Clean Label")
sns.heatmap(TM,cmap="YlGnBu", vmin=0, vmax=1,annot=True)
plt.savefig('./{}_cf.png'.format(NOISE_RATE))

noise_or_not = (clean_labels) != (noisy_labels)
index = np.arange(len(dataset.train_labels))
noise_index = index[noise_or_not]
choices = np.random.choice(noise_index,20)
plt.figure(figsize=(15,10))
plt.suptitle("IDN {} Examples".format(NOISE_RATE),fontsize=15)
for i in range(20):
    plt.subplot(4,5,i+1)
    image = dataset.train_data[choices[i]]
    plt.title('Noise Label:{}'.format(noisy_labels[choices[i]]))
    plt.imshow(image)
    plt.axis('off')
    # plt.tight_layout()
plt.savefig('./{}_IDN.png'.format(NOISE_RATE))