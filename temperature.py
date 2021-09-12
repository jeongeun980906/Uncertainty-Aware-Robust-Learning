import numpy as np
import torch
from sklearn.metrics import precision_recall_curve,roc_auc_score
import seaborn as sns
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap
import numpy as np
import os
import json
from core.network import *
from dataloader.mnist import MNIST
from dataloader.cifar import CIFAR10,CIFAR100
from torchvision import datasets,transforms

device='cuda'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def mu_eval(pi,mu,sigma,labels):
    """
        :param pi:      [N x K]
        :param mu:      [N x K x D]
        :param sigma:   [N x K x D]
    """
    max_idx = torch.argmax(pi,dim=1) # [N]
    mu      = torch.softmax(mu,dim=2) # [N x K x D]
    mu_max = torch.argmax(mu,dim=2) # [N x K]
    mu_max = torch.argmax(mu,dim=2) # [N x K]
    idx1_gather = max_idx.unsqueeze(dim=-1).repeat(1,mu.shape[2]).unsqueeze(1) # [N x 1 x D]
    mu_sel = torch.gather(mu,dim=1,index=idx1_gather).squeeze(dim=1) # [N x D]

    mu_onehot=_to_one_hot(mu_max,labels)
    mu1 = mu/0.1
    mu2 = mu/0.05

    mu1     = torch.softmax(mu1,dim=2) # [N x K x D]
    mu2     = torch.softmax(mu2,dim=2) # [N x K x D]

    pi_usq = torch.unsqueeze(pi,2) # [N x K x 1]
    pi_exp = pi_usq.expand_as(sigma) # [N x K x D]

    mu_exp1 = torch.mul(pi_exp,mu1) # mixtured mu [N x K x D]
    mu_prime1 = torch.sum(mu_exp1,dim=1) # [N x D]

    mu_exp2 = torch.mul(pi_exp,mu2) # mixtured mu [N x K x D]
    mu_prime2 = torch.sum(mu_exp2,dim=1) # [N x D]

    mu_exp = torch.mul(pi_exp,mu_onehot) # mixtured mu [N x K x D]
    mu_prime = torch.sum(mu_exp,dim=1) # [N x D]
    out = {'mu_prime1':mu_prime1,
            'mu_prime2':mu_prime2,
            'mu_prime':mu_prime,
            'mu_sel':mu_sel
           }
    return out

def plot_tm(out,data,mode,ER,idx,gt):
    DIR1='./res/rate_res/{}_{}_{}/{}_test_mu2.png'.format(data,mode,ER,idx)
    img1=out['D1']
    img2=out['D2']
    img3=out['D3']
    for i in range(10):
        img1[i,:]=img1[i,:]/(np.sum(img1[i,:]))
        img2[i,:]=img2[i,:]/(np.sum(img2[i,:]))
        img3[i,:]=img3[i,:]/(np.sum(img3[i,:]))
    img1=np.round(img1,2)
    img2=np.round(img2,2)
    img3=np.round(img3,2)
    img7=np.round(gt,2)
    NAME= data.upper()
    rate = int(100*ER)
    mode = mode.capitalize()
    if data != 'cifar100':
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('{} {} {}\% Transition Matrix Temperature\n'.format(NAME,mode,rate),fontsize=35)
        ax1.set_title('temparature: 0.1',fontsize=30)
        img = sns.heatmap(img1,ax=ax1,annot=True)
        ax1.set_xlabel('clean label',fontsize=25)
        ax1.set_ylabel('predicted label',fontsize=25)
        
        ax2.set_title('temparature: 0.05',fontsize=30)
        img = sns.heatmap(img2,ax=ax2,annot=True)
        ax2.set_xlabel('clean label',fontsize=25)
        ax2.set_ylabel('predicted label',fontsize=25)

        ax3.set_title('temparature: 0',fontsize=30)
        img = sns.heatmap(img3,ax=ax3,annot=True)
        ax3.set_xlabel('clean label',fontsize=25)
        ax3.set_ylabel('predicted label',fontsize=25)

        ax4.set_title('True',fontsize=30)
        img = sns.heatmap(img7,ax=ax4,annot=True)
        ax4.set_xlabel('clean label',fontsize=25)
        ax4.set_ylabel('predicted label',fontsize=25)
        plt.tight_layout()
    else:
        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2, 2, figsize=(16, 16))
        fig.suptitle('{} {} {}\% Transition Matrix Temperature\n'.format(Name,mode,rate),fontsize=35)
        ax1.set_title('temparature: 0.1',fontsize=30)
        img = sns.heatmap(img1,ax=ax1)
        ax1.set_xlabel('clean label',fontsize=25)
        ax1.set_ylabel('predicted label',fontsize=25)
        
        ax2.set_title('temparature: 0.05',fontsize=30)
        img = sns.heatmap(img2,ax=ax2)
        ax2.set_xlabel('clean label',fontsize=25)
        ax2.set_ylabel('predicted label',fontsize=25)

        ax3.set_title('temparature: 0',fontsize=30)
        img = sns.heatmap(img3,ax=ax3)
        ax3.set_xlabel('clean label',fontsize=25)
        ax3.set_ylabel('predicted label',fontsize=25)

        ax4.set_title('True',fontsize=30)
        img = sns.heatmap(img7,ax=ax4)
        ax4.set_xlabel('clean label',fontsize=25)
        ax4.set_ylabel('predicted label',fontsize=25)
        plt.tight_layout()
    fig.savefig(DIR1)

def mu_tm_eval(model,data_iter,data_size,device,labels):
    #false_label=torch.tensor([10]).to(device)
    with torch.no_grad():
        model.eval() # evaluate (affects DropOut and BN)
        D1=np.zeros((labels,labels))
        D2=np.zeros((labels,labels))
        D3=np.zeros((labels,labels))
        for batch_in,batch_out in data_iter:
            # Foraward path
            if data_size is None:
                mln_out     = model.forward(batch_in.to(device))
            else:
                mln_out     = model.forward(batch_in.view(data_size).to(device))
            pi,mu,sigma = mln_out['pi'],mln_out['mu'],mln_out['sigma']
            out = mu_eval(pi,mu,sigma,labels)
            mu_sel=out['mu_sel']
            mu_prime1=out['mu_prime1']
            mu_prime2=out['mu_prime2']
            mu_prime=out['mu_prime']
            _,y=torch.max(mu_sel,dim=-1)
            for i in range(batch_in.size(0)):
                D1[y[i].item()] +=mu_prime1[i].cpu().numpy()
                D2[y[i].item()] +=mu_prime2[i].cpu().numpy()
                D3[y[i].item()] +=mu_prime[i].cpu().numpy()
        model.train() # back to train mode 
        out_eval = {'D1':D1,'D2':D2,'D3':D3}
    return out_eval

def _to_one_hot(y, num_classes):
    scatter_dim = len(y.size())
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype).to(device)
        
    return zeros.scatter(scatter_dim, y_tensor, 1)

DATA = 'mnist'
noise = 'symmetric'
rate = 0.8
idx=3

if DATA == 'cifar10':
    MLN =  MixtureLogitNetwork_cnn(name='mln',x_dim=[3,32,32],c_dims = [64,64,128,128,196,16],h_dims=[],
                            p_sizes= [2,2,2], k_size=3,y_dim=10,USE_BN=True,k=24,
                            sig_min=1,sig_max=10, 
                            mu_min=-3,mu_max=+3,SHARE_SIG=True).to(device)
    data_size=(-1,3,32,32)
    transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.49137255, 0.48235294, 0.44666667), (0.24705882, 0.24352941, 0.26156863)),
        ])
    test = CIFAR10(root='./data/',download=True,train=False,transform=transform_test,
                        noise_type=noise,noise_rate=rate,test_noisy=True)
    test_iter = torch.utils.data.DataLoader(test,batch_size=128,shuffle=True,num_workers=0)
    
elif DATA == 'mnist':
    MLN = MixtureLogitNetwork_cnn2(name='mln',x_dim=[1,28,28],k_size=3,c_dims=[32,64,128],p_sizes=[2,2,2],
                            h_dims=[128,64],y_dim=10,USE_BN=False,k=10,
                            sig_min=1.0,sig_max=10, 
                            mu_min=-1,mu_max=+1,SHARE_SIG=True).to(device)
    data_size=(-1,1,28,28)
    test = MNIST(root='./data/',download=True,train=False,transform=transforms.ToTensor(),
                        noise_type=noise,noise_rate=rate,test_noisy=True)
    test_iter = torch.utils.data.DataLoader(test,batch_size=128,shuffle=True,num_workers=0)
state_dict=torch.load('./ckpt/normal/{}_{}_{}/MLN_{}.pt'.format(DATA,noise,str(rate),idx))
MLN.load_state_dict(state_dict)
out = mu_tm_eval(MLN,test_iter,data_size,device,10)
gt = test.actual_noise_rate
plot_tm(out,DATA,noise,rate,idx,gt)