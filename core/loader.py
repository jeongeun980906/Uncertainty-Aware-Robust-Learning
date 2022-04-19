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
from core.network import *
from core.summary import *

from dataloader.mnist import MNIST
from dataloader.cifar import CIFAR10,CIFAR100
from dataloader.cifar2 import cifar_dataset
from dataloader.trec import TREC
from dataloader.dirty_cifar import dirtyCIFAR10,cleanCIFAR10,ambiguousCIFAR10
from dataloader.clothing1M import *
from dataloader.dirty_mnist import DirtyMNIST,FastMNIST,AmbiguousMNIST
from core.backbones.sentence_cnn import SentenceCNN
import torchvision.models as models

def build_model(args,device,trec_config=None):
    DATASET=args.data
    if DATASET=='mnist' or DATASET=='dirty_mnist':
        model = MixtureLogitNetwork_cnn2(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            sigma = args.sigma,h_dims=[128,64],y_dim=10,USE_BN=False,k=args.k,
                            sig_min=1.0,sig_max=10, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(device)
        summary_str,summary = summary_string(model,input_size=(1,28,28),device=device)
        print("network")
        print (summary_str)
        model.init_param()
    elif DATASET == 'cifar10' or DATASET=='cifar100' or DATASET=='dirty_cifar10':
        labels=100 if DATASET=='cifar100' else 10
        if args.resnet:
            model= MixtureLogitNetwork_resnet(name='mln',x_dim=[3,32,32],
                            sigma = args.sigma,k=args.k,y_dim = labels,
                            sig_min=args.sig_min,sig_max=args.sig_max, 
                            mu_min=-3,mu_max=+3,SHARE_SIG=True).to(device)
        else:
            model= MixtureLogitNetwork_cnn(name='mln',x_dim=[3,32,32],c_dims = [64,64,128,128,196,16],h_dims=[],
                                p_sizes= [2,2,2], k_size=3,y_dim=labels,USE_BN=True,k=args.k,
                                sig_min=args.sig_min,sig_max=args.sig_max, sigma = args.sigma,
                                mu_min=-3,mu_max=+3,SHARE_SIG=True).to(device)

        # summary_str,summary = summary_string(model,input_size=(3,32,32),device=device)
        # print("network")
        # print (summary_str)
        model.init_param()
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
    elif DATASET == 'clothing1m':
        # model= MixtureLogitNetwork_resnet(name='mln',x_dim=[3,224,224],
        #                     sigma = args.sigma,k=args.k,y_dim =labels,
        #                     sig_min=args.sig_min,sig_max=args.sig_max, 
        #                     mu_min=-3,mu_max=+3,SHARE_SIG=True).to(device)
        model =  models.resnet50(pretrained=True)
        model.fc = MixtureOfLogits(in_dim=2048,y_dim=14,k=args.k,
                        sig_min=args.sig_min,sig_max=args.sig_max, sigma = args.sigma,SHARE_SIG=True)
        model.fc.fc_mu.bias.data.uniform_(-3,3)
        model.fc.fc_pi.bias.data.uniform_(-0.001,0.001)
        model = model.to(device)
    return model

def build_dataset(args):
    DATASET=args.data
    transition_matrix=None
    if DATASET=='mnist':
        input_size=(-1,1,28,28)
        num_classes=10
        if args.mode not in ['symmetric','asymmetric','fairflip','mixup','instance']:
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
                        noise_type=args.mode,noise_rate=args.ER,test_noisy=True)
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
        if args.mode not in ['symmetric','asymmetric','fairflip','clean','instance']:
            num=int(args.mode[-1])
            train = CIFAR10(root='./data/',download=True,train=True,transform=transform_train,
                        noise_type='asymmetric2',noise_rate=args.ER,num=num)
            val = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type='asymmetric2',noise_rate=args.ER,test_noisy=False,num=num)
            test = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type='asymmetric2',noise_rate=args.ER,test_noisy=True,num=num)
        else:
            # train = cifar_dataset(dataset = 'cifar10',root_dir='./data/cifar-10-batches-py'
            #             ,mode='all',transform=transform_train,noise_file = './data/cifar10_noise_file_{}_{}'.format(args.mode,args.ER),
            #             noise_mode=args.mode,r=args.ER)
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
        train_iter,val_iter,test_iter,transition_matrix,_=trec.load_trec(args.cross_validation)
        initW = trec.embedding()
        num=1
        trec_config={
            "initW":initW,"filter_list":filter_list,"filter_counts":filter_counts
        }
        input_size=None
        num_classes=6
    
    elif DATASET=='clothing1m':
        input_size=(-1,3,224,224)
        num_classes=14
        transition_matrix = None
        num=2

        transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ])    
        transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])  
        


        train = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_train,
            mode        = 'train',
            num_samples = 1000*args.batch_size
        )

        val = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_val,
            mode        = 'val'
        )
        test = clothing1M(
            root        = '/home/sungjoon.choi/seungyoun/Clothing1M',
            transform   = transform_val,
            mode        = 'test'
        )
        

    elif DATASET=='dirty_mnist':
        train = DirtyMNIST("./data/", train=True, download=True, device="cuda",noise_type=args.mode,noise_rate=args.ER)
        val = DirtyMNIST("./data/", train=False, download=True, device="cuda",noise_type='clean',noise_rate=args.ER,test_noisy=False)
        test=None
        clean_test = FastMNIST("./data/",train=False, download=True,device='cuda')#,noise_type=noise_type,noise_rate=noise_rate)
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
                                                        noise_type=args.mode,test_noisy=False,noise_rate=args.ER,alpha=args.alpha)
        test=None
        clean_test = cleanCIFAR10("./data/",download=True,train=False,transform=transform_test)
        ambiguous_test = ambiguousCIFAR10("./data/", train=False, download=True, transform=transform_test,test_noisy=True,
                                                    noise_type=args.mode,noise_rate=args.ER,alpha=args.alpha)

    if DATASET != 'trec':
        BATCH_SIZE = args.batch_size
        if args.cross_validation:
            train_iter = torch.utils.data.DataLoader(train,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
        else:
            train_iter = torch.utils.data.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
        val_iter = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE,shuffle=False,num_workers=4)
        if test==None:
            transition_matrix=ambiguous_test.transition_matrix
            ambiguous_test_iter = torch.utils.data.DataLoader(
                ambiguous_test,
                batch_size=128,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            clean_test_iter = torch.utils.data.DataLoader(
                clean_test,
                batch_size=128,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
            )
            test_iter=[ambiguous_test_iter,clean_test_iter]
        else:
            transition_matrix=test.transition_matrix
            test_iter = [torch.utils.data.DataLoader(test,batch_size=BATCH_SIZE,shuffle=False,num_workers=0)]
        trec_config=None
    else:
        test_iter = [test_iter]
    config={
        "input_size":input_size,"num_classes":num_classes,"num":num,
        'transition_matrix':transition_matrix,'val_noise_rate':None
    }
    return train_iter,val_iter,test_iter,config,trec_config


def get_cross_val_dataset(args,val_idx):
    DATASET=args.data
    BATCH_SIZE = 128
    if DATASET=='mnist':
        train = MNIST(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                    noise_type=args.mode,noise_rate=args.ER,indicies=None)
        val = MNIST(root='./data/',download=True,train=True,transform=transforms.ToTensor(),
                    noise_type=args.mode,noise_rate=args.ER,indicies=val_idx)
        actual_noise_rate = val.actual_noise_rate
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

        train = CIFAR10(root='./data/',download=True,train=True,transform=transform_train,
                        noise_type=args.mode,noise_rate=args.ER)
        val = CIFAR10(root='./data/',download=True,train=True,transform=transform_test,
                        noise_type=args.mode,noise_rate=args.ER,indicies=val_idx)
        actual_noise_rate = val.actual_noise_rate
    elif DATASET=='cifar100':
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
                    noise_type=args.mode,noise_rate=args.ER,test_noisy=False,indicies=val_idx)
        actual_noise_rate = val.actual_noise_rate
    elif DATASET=='trec':
        filter_list = [3,4,5]
        filter_counts= [100]*len(filter_list)

        trec=TREC(noise_type=args.mode,noise_rate=args.ER,batch_size=50,test_sample_percentage=0.1,filter_list = filter_list,IB=args.sampler)
        train_iter, val_iter, _, _ , actual_noise_rate= trec.load_trec(False,val_idx)
        return train_iter,val_iter,actual_noise_rate

    train_iter = torch.utils.data.DataLoader(train,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    val_iter = torch.utils.data.DataLoader(val,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
    return train_iter,val_iter,actual_noise_rate