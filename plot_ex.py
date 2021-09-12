from os import access
from core.plot_MLN import plot_res
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

TYPE=['symmetric_0.2','symmetric_0.5','symmetric_0.8','asymmetric_0.4']#,'asymmetric2_0.4','asymmetric3_0.6']
ID=[6]*4
transition,gt,pi,dpi={},{},{},{}
mode=1
DATA='mnist'
labels=10

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
def plot_tm(transition,gt):
    fig, axes=plt.subplots(4,4,figsize=(22,22))
    axes[0][0].text(0.5, 0.5, 'Predicted', fontsize=29,verticalalignment='center')
    axes[1][0].text(0.5, 0.5, 'True', fontsize=29,verticalalignment='center')
    axes[0][0].axis("off")
    axes[1][0].axis("off")
    axes[2][0].text(0.5, 0.5, 'Predicted', fontsize=29,verticalalignment='center')
    axes[3][0].text(0.5, 0.5, 'True', fontsize=29,verticalalignment='center')
    axes[2][0].axis("off")
    axes[3][0].axis("off")
    NAME=DATA.upper()
    plt.suptitle("{} Label Noise Transition Matrix\n".format(NAME),fontsize=32)
    for i in range(3):
        t=TYPE[i]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        if noise != 'Asymmetric' and noise != 'Symmetric':
            if int(noise[-1])==2:
                noise = 'Dual'
            elif int(noise[-1])==3:
                noise = 'Tridiagonal'
        axes[0][i+1].set_title('{} {}\%'.format(noise,RATE),fontsize=29)
        img = sns.heatmap(transition[t],ax=axes[0][i+1],annot=True)
        axes[0][i+1].set_xlabel('Clean Label',fontsize=24)
        axes[0][i+1].set_ylabel('Noisy Label',fontsize=24)
        axes[0][i+1].tick_params(axis='x', labelsize= 20)
        #plt.tight_layout()
        
        img = sns.heatmap(gt[t],ax=axes[1][i+1],annot=True)
        axes[1][i+1].set_xlabel('Clean Label',fontsize=24)
        axes[1][i+1].set_ylabel('Noisy Label',fontsize=24)
        axes[1][i+1].tick_params(axis='x', labelsize= 20)
        #plt.tight_layout()

        t=TYPE[i+3]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        if noise != 'Asymmetric' and noise != 'Symmetric':
            if int(noise[-1])==2:
                noise = 'Dual'
            elif int(noise[-1])==3:
                noise = 'Tridiagonal'
        axes[2][i+1].set_title('{} {}\%'.format(noise,RATE),fontsize=29)
        img = sns.heatmap(transition[t],ax=axes[2][i+1],annot=True)
        axes[2][i+1].set_xlabel('Clean Label',fontsize=24)
        axes[2][i+1].set_ylabel('Noisy Label',fontsize=24)
        axes[2][i+1].tick_params(axis='x', labelsize= 20)
        #plt.tight_layout()

        img = sns.heatmap(gt[t],ax=axes[3][i+1],annot=True)
        axes[3][i+1].set_xlabel('Clean Label',fontsize=24)
        axes[3][i+1].set_ylabel('Noisy Label',fontsize=24)
        axes[3][i+1].tick_params(axis='x', labelsize= 20)
    plt.tight_layout()
    plt.savefig('./res/{}_tm.png'.format(DATA))

for i,type in enumerate(TYPE):
    DIR_PATH='./res/rate_res/{}_{}/{}_test.json'.format(DATA,TYPE[i],ID[i])
    with open(DIR_PATH,'r') as json_file:
        data=json.load(json_file)
    transition[type]=data['TM']
    gt[type]=data['GT']
    pi[type]=data['pi']
    dpi[type]=data['dpi']
plot_tm(transition,gt)
