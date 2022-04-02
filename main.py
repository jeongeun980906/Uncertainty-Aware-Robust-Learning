import os
import argparse
import torch
import random
import numpy as np
from core.solver import load,train,test
from core.cross_validation import cross_validation 

def main(args):
    torch.cuda.set_device(args.gpu)
    print(args.gpu)
    device = torch.device('cuda:{}'.format(args.gpu))
    #os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu) 
    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.cuda.manual_seed_all(0)  # GPU seed
    torch.manual_seed(seed=0)
    np.random.seed(seed=0)
    random.seed(0)
    # init Solver
    if args.cross_validation:
        cv = cross_validation(args)
        cv.load_total_dataset()
        cv.train_init()
        torch.cuda.manual_seed_all(0)  # GPU seed
        torch.manual_seed(seed=0)
        np.random.seed(seed=0)
        random.seed(0)
        cv.load_new_dataset()
        cv.gain_traisiton_matrix()
        cv.train_full()
    else:
        train_iter,val_iter,test_iter,MLN,config,dataset_config = load(args)
        if args.eval:
            test(args,test_iter,MLN,config,dataset_config)
        else:
            train(args,train_iter,val_iter,test_iter,MLN,config,dataset_config)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training mode and dataset
    parser.add_argument('--eval',action='store_true' ,default=False,help='train or test')
    parser.add_argument('--data', type=str,default='mnist',help='dataset',choices=["mnist",'cifar10','cifar100','trec','dirty_cifar10','dirty_mnist','clothing1m'])
    parser.add_argument('--sampler', type=bool,default=False,help='for imbalanced dataset')
    parser.add_argument('--batch_size', type=int,default=128,help='batch size') 
    # Noise Type
    parser.add_argument('--mode', type=str,default='symmetric',choices = ['symmetric','asymmetric','clean','instance'],help='Noise Pattern')
    parser.add_argument('--ER', type=float,default=0.2,help='Noise Rate')
    
    # parser.add_argument('--none',default=False,action = 'store_true',help='do not use mixtures') 
    # MLN Properties
    parser.add_argument('--k', type=int,default=5,help='number of mixtures') 
    parser.add_argument('--sigma',default=True,action='store_false',help='use sigma attenuation term')
    parser.add_argument('--sig_max', type=float,default=10,help='sig_max')
    parser.add_argument('--sig_min', type=float,default=1,help='sig_min')
    parser.add_argument('--ratio', type=float,default=1,help='ratio for epis')
    parser.add_argument('--ratio2', type=float,default=1,help='ratio for alea')
    
    # Hyperparameters for learning
    parser.add_argument('--resnet', default=False,action='store_true',help='for resnet')
    parser.add_argument('--lr', type=float,default=1e-3,help='learing rate')
    parser.add_argument('--wd', type=float,default=5e-4,help='weight decay')
    parser.add_argument('--lr_rate', type=float,default=0.2,help='learing rate schedular rate')
    parser.add_argument('--lr_step', type=int,default=20,help='learing rate schedular step')
    parser.add_argument('--epoch', type=int,default=200,help='epoch')

    # Adaptive ratio schedular
    parser.add_argument('--tunner', type=int,default=0,help='adaptive ratio schedular')
    
    # For dirtycifar10 cutmix rate
    parser.add_argument('--mixup', default=False,action='store_true',help='for mixup')
    parser.add_argument('--alpha', type=float,default=0.5,help='for mixing')
    
    # GPU config
    parser.add_argument('--gpu', type=int,default=0,help='gpu device')
    
    # save index
    parser.add_argument('--id', type=int,default=1,help='save index')

    parser.add_argument('--cross_validation', type=int,default=0,help='cross validation')
    args = parser.parse_args()
    main(args)
