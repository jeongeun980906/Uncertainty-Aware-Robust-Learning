import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as TD
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets,transforms

from wandb.wandb_controller import sweep
from core.network import *
from core.summary import *
from dataloader.mnist import MNIST
from dataloader.cifar import CIFAR10,CIFAR100
from dataloader.trec import TREC
from dataloader.dirty_cifar import dirtyCIFAR10,cleanCIFAR10,ambiguousCIFAR10
from dataloader.dirty_mnist import DirtyMNIST,FastMNIST,AmbiguousMNIST
from core.backbones.sentence_cnn import SentenceCNN

def build_model(args,device,trec_config=None):
    DATASET=args.data
    if DATASET=='mnist' or DATASET=='dirty_mnist':
        model = MixtureLogitNetwork_cnn2(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            h_dims=[128,64],y_dim=10,USE_BN=False,k=args.k,
                            sig_min=1.0,sig_max=10, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(device)
        summary_str,summary = summary_string(model,input_size=(1,28,28),device=device)
        print("network")
        print (summary_str)
    elif DATASET == 'cifar10' or DATASET=='cifar100' or DATASET=='dirty_cifar10':
        labels=100 if DATASET=='cifar100' else 10
        model= MixtureLogitNetwork_cnn(name='mln',x_dim=[3,32,32],c_dims = [64,64,128,128,196,16],h_dims=[],
                            p_sizes= [2,2,2], k_size=3,y_dim=labels,USE_BN=True,k=args.k,
                            sig_min=args.sig_min,sig_max=args.sig_max, 
                            mu_min=-3,mu_max=+3,SHARE_SIG=True).to(device)
        summary_str,summary = summary_string(model,input_size=(3,32,32),device=device)
        print("network")
        print (summary_str)
    elif DATASET == 'trec':
        textcnn = SentenceCNN(nb_classes=6,
                        word_embedding_numpy=trec_config["initW"],
                        filter_lengths=trec_config["filter_list"],
                        filter_counts=trec_config["filter_counts"],
                        dropout_rate=0.5).to(device)
        model = MixtureLogitNetwork_TextCNN(y_dim      = 6,       # output dimension
                                USE_BN     = True,            # whether to use batch-norm
                                k          = args.k,               # number of mixtures
                                sig_min    = args.sig_min,    # minimum sigma
                                sig_max    = args.sig_max,    # maximum sigma
                                mu_min     = -2,              # minimum mu (init)
                                mu_max     = +2,              # maximum mu (init)
                                model      = textcnn,
                                filter_counts = trec_config["filter_counts"],
                                SHARE_SIG  = True).to(device)
    model.init_param()
    return model

def build_dataset(args):
    DATASET=args.data
    transition_matrix=None
    if DATASET=='mnist':
        input_size=(-1,1,28,28)
        num_classes=10
        if args.mode not in ['symmetric','asymmetric','fairflip','mixup']:
            num=int(args.mode[-1])
            train = MNIST(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                        noise_type='asymmetric2',noise_rate=args.ER,num=num)
            val = MNIST(root='./data/',download=True,train=False,transform=transforms.ToTensor(),
                        noise_type=args.mode,noise_rate=args.ER,num=num)
            test = MNIST(root='./data/',download=True,train=False,transform=transforms.ToTensor(),
                        noise_type=args.mode,noise_rate=args.ER,num=num,test_noisy=True)
        else:
            train = MNIST(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                        noise_type=args.mode,noise_rate=args.ER)
            val = MNIST(root='./data/',download=True,train=False,transform=transforms.ToTensor(),
                        noise_type=args.mode,noise_rate=args.ER)
            test = MNIST(root='./data/',download=True,train=False,transform=transforms.ToTensor(),
                        noise_type=args.mode,noise_rate=args.ER)
            num=1
    elif DATASET=='cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])
        input_size=(-1,3,32,32)
        num_classes=10
        if args.mode not in ['symmetric','asymmetric','fairflip']:
            num=int(args.mode[-1])
            train = CIFAR10(root='./data/',download=True,train=True,transform=transform_train,
                        noise_type='asymmetric2',noise_rate=args.ER,num=num)
            val = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type='asymmetric2',noise_rate=args.ER,test_noisy=False,num=num)
            test = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type='asymmetric2',noise_rate=args.ER,test_noisy=True,num=num)
        else:
            train = CIFAR10(root='./data/',download=True,train=True,transform=transform_train,
                        noise_type=args.mode,noise_rate=args.ER)
            val = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type=args.mode,noise_rate=args.ER,test_noisy=False)
            test = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type=args.mode,noise_rate=args.ER,test_noisy=True)
            num=1
    elif DATASET=='cifar100':
        input_size=(-1,3,32,32)
        num_classes=100
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])
        train = CIFAR100(root='./data/',download=True,train=True,transform=transform_train,
                    noise_type=args.mode,noise_rate=args.ER)
        val = CIFAR100(root='./data/',download=True,train=False,transform=transform_test,
                    noise_type=args.mode,noise_rate=args.ER,test_noisy=False)
        test = CIFAR100(root='./data/',download=True,train=False,transform=transform_test,
                    noise_type=args.mode,noise_rate=args.ER,test_noisy=True)
        num=1
    
    elif DATASET=='trec':
        filter_list = [3,4,5]
        filter_counts= [100]*len(filter_list)

        trec=TREC(noise_type=args.mode,noise_rate=args.ER,batch_size=50,test_sample_percentage=0.1,filter_list = filter_list,IB=args.sampler)
        train_iter,val_iter,test_iter,transition_matrix=trec.load_trec()
        initW = trec.embedding()
        num=1
        trec_config={
            "initW":initW,"filter_list":filter_list,"filter_counts":filter_counts
        }
        input_size=None
        num_classes=6
        
    elif DATASET=='dirty_mnist':
        train = DirtyMNIST("./data/", train=True, download=True, device="cuda",noise_type=args.mode,noise_rate=args.ER)
        val = DirtyMNIST("./data/", train=False, download=True, device="cuda",noise_type=args.mode,noise_rate=args.ER)
        test=None
        clean_test = FastMNIST("./data/",download=True,device='cuda')#,noise_type=noise_type,noise_rate=noise_rate)
        ambiguous_test = AmbiguousMNIST("./data/", train=False, download=True, device="cuda",noise_type=args.mode,noise_rate=args.ER)
        num=1
        input_size=(-1,1,28,28)
        num_classes=10

    elif DATASET=='dirty_cifar10':
        input_size=(-1,3,32,32)
        num_classes=10
        num=1
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])

        train = dirtyCIFAR10("./data/", train=True, download=True,transform=transform_train,
                                                        noise_type=args.mode,noise_rate=args.ER,alpha=args.alpha)
        val = dirtyCIFAR10("./data/", train=False, download=True,transform=transform_test,
                                                        noise_type=args.mode,noise_rate=args.ER,alpha=args.alpha)
        test=None
        clean_test = cleanCIFAR10("./data/",download=True,train=False,transform=transform_test)
        ambiguous_test = ambiguousCIFAR10("./data/", train=False, download=True, transform=transform_test,test_noisy=True,
                                                    noise_type=args.mode,noise_rate=args.ER,alpha=args.alpha)

    if DATASET != 'trec':
        BATCH_SIZE = 128
        train_iter = torch.utils.data.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
        val_iter = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
        if test==None:
            transition_matrix=ambiguous_test.actual_noise_rate
            ambiguous_test_iter = torch.utils.data.DataLoader(
                ambiguous_test,
                batch_size=128,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            clean_test_iter = torch.utils.data.DataLoader(
                clean_test,
                batch_size=128,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            test_iter=[ambiguous_test_iter,clean_test_iter]
        else:
            transition_matrix=train.actual_noise_rate
            test_iter = [torch.utils.data.DataLoader(test,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)]
        trec_config=None
    else:
        test_iter = [test_iter]
    config={
        "input_size":input_size,"num_classes":num_classes,"num":num,'transition_matrix':transition_matrix
    }
    return train_iter,val_iter,test_iter,config,trec_config