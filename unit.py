from dataloader.cifar import CIFAR10,CIFAR100
from torchvision import datasets,transforms

train = CIFAR10(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                    noise_type='symmetric',noise_rate=0.2,indicies=[1,2,3])
                    