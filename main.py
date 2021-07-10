import os
import argparse

import torch
import random
import numpy as np
import wandb
from core.solver import load,train,test

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
    train_iter,val_iter,test_iter,MLN,config,dataset_config = load(args)
    if args.train:
        train(args,train_iter,val_iter,test_iter,MLN,config,dataset_config)
    else:
        test(args,train_iter,val_iter,test_iter,MLN,config,dataset_config)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Training mode and dataset
    parser.add_argument('--train', type=int,default=1,help='train or test')
    parser.add_argument('--data', type=str,default='mnist',help='dataset',choices=["mnist",'cifar10','cifar100','trec','dirty_cifar10','dirty_mnist'])
    parser.add_argument('--sampler', type=bool,default=False,help='for imbalanced dataset')

    # Noise Type
    parser.add_argument('--mode', type=str,default='symmetric',help='Noise Pattern')
    parser.add_argument('--ER', type=float,default=0.2,help='Noise Rate')
    
    # MLN Properties
    parser.add_argument('--k', type=int,default=5,help='number of mixtures') 
    parser.add_argument('--sig_max', type=float,default=10,help='sig_max')
    parser.add_argument('--sig_min', type=float,default=1,help='sig_min')
    parser.add_argument('--ratio', type=float,default=1,help='ratio for epis')
    parser.add_argument('--ratio2', type=float,default=1,help='ratio for alea')
    
    # Hyperparameters for learning
    parser.add_argument('--lr', type=float,default=1e-3,help='learing rate')
    parser.add_argument('--wd', type=float,default=5e-6,help='weight decay')
    parser.add_argument('--lr_step', type=list,default=[50,100,150],help='learing rate schedular')
    parser.add_argument('--lr_rate', type=float,default=0.5,help='learing rate schedular rate')
    parser.add_argument('--epoch', type=int,default=200,help='epoch')

    # Adaptive ratio schedular
    parser.add_argument('--ratio_schedular', type=int,default=0,help='adaptive ratio schedular')
    
    # For dirtycifar10 cutmix rate
    parser.add_argument('--alpha', type=float,default=0.5,help='for mixing')
    
    # GPU config
    parser.add_argument('--gpu', type=int,default=0,help='gpu device')
    
    # save index
    parser.add_argument('--id', type=int,default=1,help='save index')
    
    # wandb
    parser.add_argument('--wandb', type=int,default=0,help='use wandb')
    parser.add_argument('--sweep', type=int,default=0,help='use wandb sweep')

    args = parser.parse_args()

    # W&B hyperparam sweep
    if args.train:
        print('train mode')
        if args.sweep:
            if args.data=='cifar100':
                param_dict = {
                'ratio1': {
                    'values': [1,0.5,0.1,0.01,0.05,0.001,0.0]
                },
                'ratio2': {
                    'values': [0.01]
                }
            }
            else:
                param_dict = {
                'ratio1': {
                    'values': [0.1,5,10,1,0.5,0.0]
                },
                'ratio2': {
                    'values': [0.1]
                }
            }
            sweep_config = {
                'name': '{}_{}_{}'.format(args.data,args.mode,args.ER),
                'method': 'grid', #grid, random
                # 'method': 'random', #grid, random
                'metric': {
                    'name': 'loss',
                    'goal': 'minimize'   
                },
                'parameters': param_dict
            }
            sweep_id = wandb.sweep(sweep_config, 
                                project='MLN_sweep_{}_{}_{}'.format(args.data,args.mode,args.ER)
                                #    project="rstruc-sweep"
                                )
            wandb.agent(sweep_id=sweep_id, function=lambda:main(args),count=20)
            # wandb.agent(sweep_id=sweep_id, function=lambda:main(args), entity="jtk", count=10) # limit max num run (for random sweep)
        else:
            main(args)
    else:
        print('test mode')
        main(args)