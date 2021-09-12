from os import access
from core.plot_MLN import plot_res
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns

TYPE=['symmetric_0.2','symmetric_0.5','symmetric_0.8','asymmetric_0.4']
ID=[3]*4
transition_clean,gt_clean,pi_clean,dpi_clean={},{},{},{}
transition_amb,gt_amb,pi_amb,dpi_amb={},{},{},{}
alea_amb,alea_clean={},{}
mode=1
DATA='dirty_mnist'
labels=10

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
def plot_tm():
    fig, axes=plt.subplots(4,5,figsize=(26,24))#}) ,gridspec_kw={'width_ratios': [1,3,3,3,3]
    axes[0][0].text(0.5, 0.5, 'Predicted', fontsize=32,verticalalignment='center')
    axes[1][0].text(0.5, 0.5, 'True', fontsize=32,verticalalignment='center')
    axes[0][0].axis("off")
    axes[1][0].axis("off")
    axes[2][0].text(0.5, 0.5, 'Predicted', fontsize=32,verticalalignment='center')
    axes[3][0].text(0.5, 0.5, 'True', fontsize=32,verticalalignment='center')
    axes[2][0].axis("off")
    axes[3][0].axis("off")
    NAME = DATA[6:]
    NAME=NAME.upper()
    plt.suptitle("Dirty {} Label Noise Transition Matrix\n\n\n".format(NAME),fontsize=36)
    plt.figtext(0.25,0.94,"Clean",fontsize=32)
    plt.figtext(0.45,0.94,"Ambiguous",fontsize=32)
    plt.figtext(0.65,0.94,"Clean",fontsize=32)
    plt.figtext(0.85,0.94,"Ambiguous",fontsize=32)
    # plt.figtext(0.2,0.94,"Clean",fontsize=28)
    # plt.figtext(0.4,0.94,"Ambiguous",fontsize=28)
    # plt.figtext(0.65,0.94,"Clean",fontsize=28)
    # plt.figtext(0.85,0.94,"Ambiguous",fontsize=28)
    for i in range(2):
        t=TYPE[i]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        #axes[0][2*i+1].set_title('{} {}\%'.format(noise,RATE),fontsize=28)
        if noise != 'Asymmetric' and noise != 'Symmetric':
            if int(noise[-1])==2:
                noise = 'Dual'
            elif int(noise[-1])==3:
                noise = 'Tridiagonal'
        plt.figtext(0.4*i+0.35,0.92,'{} {}\%'.format(noise,RATE),fontsize=32)
        img = sns.heatmap(transition_clean[t],ax=axes[0][2*i+1],annot=True)
        axes[0][2*i+1].set_xlabel('Clean Label',fontsize=29)
        axes[0][2*i+1].set_ylabel('Noisy Label',fontsize=29)
        axes[0][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[0][2*i+1].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        img = sns.heatmap(gt_clean[t],ax=axes[1][2*i+1],annot=True)
        axes[1][2*i+1].set_xlabel('Clean Label',fontsize=29)
        axes[1][2*i+1].set_ylabel('Noisy Label',fontsize=29)
        axes[1][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[1][2*i+1].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        img = sns.heatmap(transition_amb[t],ax=axes[0][2*i+2],annot=True)
        axes[0][2*i+2].set_xlabel('Clean Label',fontsize=29)
        axes[0][2*i+2].set_ylabel('Noisy Label',fontsize=29)
        axes[0][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[0][2*i+2].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()
        img = sns.heatmap(gt_amb[t],ax=axes[1][2*i+2],annot=True)
        axes[1][2*i+2].set_xlabel('Clean Label',fontsize=29)
        axes[1][2*i+2].set_ylabel('Noisy Label',fontsize=29)
        axes[1][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[1][2*i+2].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        t=TYPE[i+2]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        if noise != 'Asymmetric' and noise != 'Symmetric':
            if int(noise[-1])==2:
                noise = 'Dual'
            elif int(noise[-1])==3:
                noise = 'Tridiagonal'
        axes[2][i+1].set_title('\n'.format(noise,RATE),fontsize=32)
        plt.figtext(0.4*i+0.35,0.46,'{} {}\%'.format(noise,RATE),fontsize=32)
        img = sns.heatmap(transition_clean[t],ax=axes[2][2*i+1],annot=True)
        axes[2][2*i+1].set_xlabel('Clean Label',fontsize=29)
        axes[2][2*i+1].set_ylabel('Noisy Label',fontsize=29)
        axes[2][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[2][2*i+1].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        img = sns.heatmap(gt_clean[t],ax=axes[3][2*i+1],annot=True)
        axes[3][2*i+1].set_xlabel('Clean Label',fontsize=29)
        axes[3][2*i+1].set_ylabel('Noisy Label',fontsize=29)
        axes[3][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[3][2*i+1].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        img = sns.heatmap(transition_amb[t],ax=axes[2][2*i+2],annot=True)
        axes[2][2*i+2].set_xlabel('Clean Label',fontsize=29)
        axes[2][2*i+2].set_ylabel('Noisy Label',fontsize=29)
        axes[2][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[2][2*i+2].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()

        img = sns.heatmap(gt_amb[t],ax=axes[3][2*i+2],annot=True)
        axes[3][2*i+2].set_xlabel('Clean Label',fontsize=29)
        axes[3][2*i+2].set_ylabel('Noisy Label',fontsize=29)
        axes[3][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[3][2*i+2].tick_params(axis='y', labelsize= 24)
        plt.tight_layout()
    plt.savefig('./res/{}_tm.png'.format(DATA))

def plot_pi():
    fig, axes=plt.subplots(4,5,figsize=(22,16),gridspec_kw={'width_ratios': [1,3,3,3,3]})
    axes[0][0].text(0.5, 0.5, '$\pi$', fontsize=32,verticalalignment='center')
    axes[1][0].text(0.5, 0.5, '$\delta \pi$', fontsize=32,verticalalignment='center')
    axes[0][0].axis("off")
    axes[1][0].axis("off")
    axes[2][0].text(0.5, 0.5, '$\pi$', fontsize=32,verticalalignment='center')
    axes[3][0].text(0.5, 0.5, '$\delta \pi$', fontsize=32,verticalalignment='center')
    axes[2][0].axis("off")
    axes[3][0].axis("off")
    NAME = DATA[6:]
    NAME=NAME.upper()
    plt.suptitle("Dirty {} Label Noise Model\n\n\n".format(NAME),fontsize=34)
    plt.figtext(0.15,0.93,"Clean",fontsize=32,verticalalignment='center')
    plt.figtext(0.35,0.93,"Ambiguous",fontsize=32,verticalalignment='center')
    plt.figtext(0.6,0.93,"Clean",fontsize=32,verticalalignment='center')
    plt.figtext(0.8,0.93,"Ambiguous",fontsize=32,verticalalignment='center')
    for i in range(2):
        t=TYPE[i]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        if noise != 'Symmetric':
            if noise=='Asymmetric':
                n=3
            else:
                n=int(noise[-1])+2
                if int(noise[-1])==2:
                    noise = 'Dual'
                elif int(noise[-1])==3:
                    noise = 'Tridiagonal'
            if DATA=='dirty_cifar10':
                IS_NOISE=[2,3,4,5,9]
                NOT_NOISE=[0,1,6,7,8]
            elif DATA=='dirty_mnist':
                IS_NOISE = [7,2,5,6,3]
                NOT_NOISE=[0,1,4,8,9]
            else:
                IS_NOISE=[i for i in range(labels)]
                NOT_NOISE=[]
        else:
            n=3
            IS_NOISE=[i for i in range(labels)]
            NOT_NOISE=[]
        #axes[0][2*i+1].set_title('{} {}\%'.format(noise,RATE),fontsize=28)
        plt.figtext(0.4*i+0.3,0.89,'{} {}\%'.format(noise,RATE),fontsize=32,ha='center', va='center')
        axes[0][2*i+1].set_xticks([i for i in range(10)])
        axes[0][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[0][2*i+1].tick_params(axis='y', labelsize= 24)
        axes[0][2*i+1].set_xlabel('Class',fontsize=29)
        if noise =='Symmetric':
            axes[0][2*i+1].set_ylim([0.18,0.5])
        for j in range(n+1):
            axes[0][2*i+1].plot(pi_clean[t][j],label='$\pi({})$'.format(j+1),linewidth=1)
        #axes[0][2*i+1].plot(NOT_NOISE,clean2[0,:],'bo',markersize=5,label='clean label')
        #axes[0][2*i+1].plot(IS_NOISE,noisy2[0,:],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            for j in range(0,n-1):
                axes[0][2*i+1].plot(pi_clean[t][j],'bo',markersize=5)
        else:
            axes[0][2*i+1].plot(pi_clean[t][0],'bo',markersize=5)
            axes[0][2*i+1].plot(pi_clean[t][1],'bo',markersize=5)
        if i==2:
            axes[0][2*i+1].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()
        
        axes[1][2*i+1].set_xlabel('Class',fontsize=29)
        axes[1][2*i+1].set_xticks([i for i in range(labels)])
        axes[1][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[1][2*i+1].tick_params(axis='y', labelsize= 24)
        if noise =='Symmetric':
            axes[1][2*i+1].set_ylim([0.0,0.28])

        for j in range(n):
            axes[1][2*i+1].plot(dpi_clean[t][j],label='$\pi({})-\pi({})$'.format(j+1,j+2),linewidth=1)
        #axes[1][2*i+1].plot(NOT_NOISE,clean[0,:],'bo',markersize=5,label='clean label')
        #axes[1][2*i+1].plot(IS_NOISE,noisy[0,:],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            for j in range(0,n-1):
                axes[1][2*i+1].plot(dpi_clean[t][j],'bo',markersize=5)
        else:
            axes[1][2*i+1].plot(dpi_clean[t][0],'bo',markersize=5)
            axes[1][2*i+1].plot(dpi_clean[t][1],'bo',markersize=5)
        if i==2:
            axes[1][2*i+1].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()

        axes[0][2*i+2].set_xticks([i for i in range(10)])
        axes[0][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[0][2*i+2].tick_params(axis='y', labelsize= 24)
        axes[0][2*i+2].set_xlabel('Class',fontsize=29)
        if noise=='Symmetric':
           axes[0][2*i+2].set_ylim([0.18,0.5])
        clean2=np.take(pi_amb[t],NOT_NOISE,axis=1)
        noisy2=np.take(pi_amb[t],IS_NOISE,axis=1)
        for j in range(n+1):
            axes[0][2*i+2].plot(pi_amb[t][j],label='$\pi({})$'.format(j+1),linewidth=1)
        axes[0][2*i+2].plot(NOT_NOISE,clean2[0,:],'bo',markersize=5,label='clean label')
        axes[0][2*i+2].plot(IS_NOISE,noisy2[0,:],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            for j in range(1,n-1):
                axes[0][2*i+2].plot(NOT_NOISE,clean2[j,:],'bo',markersize=5)
                axes[0][2*i+2].plot(IS_NOISE,noisy2[j,:],'ro',markersize=5)
        else:
            axes[0][2*i+2].plot(NOT_NOISE,clean2[1,:],'bo',markersize=5)
            axes[0][2*i+2].plot(IS_NOISE,noisy2[1,:],'ro',markersize=5)
        if i==1:
            axes[0][2*i+2].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()
        
        axes[1][2*i+2].set_xlabel('Class',fontsize=29)
        axes[1][2*i+2].set_xticks([i for i in range(labels)])
        axes[1][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[1][2*i+2].tick_params(axis='y', labelsize= 24)
        if noise=='Symmetric':
            axes[1][2*i+2].set_ylim([0.0,0.28])
        clean=np.take(dpi_amb[t],NOT_NOISE,axis=1)
        noisy=np.take(dpi_amb[t],IS_NOISE,axis=1)
        for j in range(n):
            axes[1][2*i+2].plot(dpi_amb[t][j],label='$\pi({})-\pi({})$'.format(j+1,j+2),linewidth=1)
        axes[1][2*i+2].plot(NOT_NOISE,clean[0],'bo',markersize=5,label='clean label')
        axes[1][2*i+2].plot(IS_NOISE,noisy[0],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            axes[1][2*i+2].plot(NOT_NOISE,clean[n-2],'bo',markersize=5)
            axes[1][2*i+2].plot(IS_NOISE,noisy[n-2],'ro',markersize=5)
        else:
            axes[1][2*i+2].plot(NOT_NOISE,clean[1],'bo',markersize=5)
            axes[1][2*i+2].plot(IS_NOISE,noisy[1],'ro',markersize=5)
        if i==1:
            axes[1][2*i+2].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()

        t=TYPE[i+2]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        if noise != 'Symmetric':
            if noise=='Asymmetric':
                n=3
            else:
                n=int(noise[-1])+1
                if int(noise[-1])==2:
                    noise = 'Dual'
                elif int(noise[-1])==3:
                    noise = 'Tridiagonal'
            if DATA=='dirty_cifar10':
                IS_NOISE=[2,3,4,5,9]
                NOT_NOISE=[0,1,6,7,8]
            elif DATA=='dirty_mnist':
                IS_NOISE = [7,2,5,6,3]
                NOT_NOISE=[0,1,4,8,9]
            else:
                IS_NOISE=[i for i in range(labels)]
                NOT_NOISE=[]
        else:
            n=3
            IS_NOISE=[i for i in range(labels)]
            NOT_NOISE=[]
        axes[2][2*i+1].set_title('\n'.format(noise,RATE),fontsize=32)
        plt.figtext(0.4*i+0.3,0.44,'{} {}\%'.format(noise,RATE),fontsize=32,ha='center', va='center')
        axes[2][2*i+1].set_xticks([i for i in range(10)])
        axes[2][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[2][2*i+1].tick_params(axis='y', labelsize= 24)
        axes[2][2*i+1].set_xlabel('Class',fontsize=29)
        if noise=='Symmetric':
            axes[2][2*i+1].set_ylim([0.18,0.5])
        print(len(pi_clean[t]),n)
        for j in range(n+1):
            axes[2][2*i+1].plot(pi_clean[t][j],label='$\pi({})$'.format(j+1),linewidth=1)
        if noise != 'Symmetric':
            for j in range(0,n):
                axes[2][2*i+1].plot(pi_clean[t][j],'bo',markersize=5)
        else:
            axes[2][2*i+1].plot(pi_clean[t][0],'bo',markersize=5)
            axes[2][2*i+1].plot(pi_clean[t][1],'bo',markersize=5)
        plt.tight_layout()
        
        axes[3][2*i+1].set_xlabel('Class',fontsize=29)
        axes[3][2*i+1].set_xticks([i for i in range(labels)])
        axes[3][2*i+1].tick_params(axis='x', labelsize= 24)
        axes[3][2*i+1].tick_params(axis='y', labelsize= 24)
        if noise=='Symmetric':
            axes[3][2*i+1].set_ylim([0.0,0.28])

        for j in range(n-1):
            axes[3][2*i+1].plot(dpi_clean[t][j],label='$\pi({})-\pi({})$'.format(j+1,j+2),linewidth=1)
        if noise != 'Symmetric':
            for j in range(0,n-1):
                axes[3][2*i+1].plot(dpi_clean[t][j],'bo',markersize=5)
        else:
            axes[3][2*i+1].plot(dpi_clean[t][0],'bo',markersize=5)
            axes[3][2*i+1].plot(dpi_clean[t][1],'bo',markersize=5)
        plt.tight_layout()

        axes[2][2*i+2].set_xticks([i for i in range(10)])
        axes[2][2*i+2].set_xlabel('Class',fontsize=29)
        axes[2][2*i+2].set_xticks([j for j in range(labels)])
        axes[2][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[2][2*i+2].tick_params(axis='y', labelsize= 24)
        if noise =='Symmetric':
            axes[2][2*i+2].set_ylim([0.18,0.5])
        clean2=np.take(pi_amb[t],NOT_NOISE,axis=1)
        noisy2=np.take(pi_amb[t],IS_NOISE,axis=1)
        for j in range(n+1):
            axes[2][2*i+2].plot(pi_amb[t][j],label='$\pi({})$'.format(j+1),linewidth=1)
        axes[2][2*i+2].plot(NOT_NOISE,clean2[0,:],'bo',markersize=5,label='clean label')
        axes[2][2*i+2].plot(IS_NOISE,noisy2[0,:],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            for j in range(1,n-1):
                axes[2][2*i+2].plot(NOT_NOISE,clean2[j,:],'bo',markersize=5)
                axes[2][2*i+2].plot(IS_NOISE,noisy2[j,:],'ro',markersize=5)
        else:
            axes[2][2*i+2].plot(NOT_NOISE,clean2[1,:],'bo',markersize=5)
            axes[2][2*i+2].plot(IS_NOISE,noisy2[1,:],'ro',markersize=5)
        if i==1:
            axes[2][2*i+2].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()
        
        axes[3][2*i+2].set_xlabel('Class',fontsize=29)
        axes[3][2*i+2].set_xticks([j for j in range(labels)])
        axes[3][2*i+2].tick_params(axis='x', labelsize= 24)
        axes[3][2*i+2].tick_params(axis='y', labelsize= 24)
        if noise =='Symmetric':
            axes[3][2*i+2].set_ylim([0.0,0.28])
        clean=np.take(dpi_amb[t],NOT_NOISE,axis=1)
        noisy=np.take(dpi_amb[t],IS_NOISE,axis=1)
        for j in range(n):
            axes[3][2*i+2].plot(dpi_amb[t][j],label='$\pi({})-\pi({})$'.format(j+1,j+2),linewidth=1)
        axes[3][2*i+2].plot(NOT_NOISE,clean[0],'bo',markersize=5,label='clean label')
        axes[3][2*i+2].plot(IS_NOISE,noisy[0],'ro',markersize=5,label='noisy label')
        if noise != 'Symmetric':
            axes[3][2*i+2].plot(NOT_NOISE,clean[n-2],'bo',markersize=5)
            axes[3][2*i+2].plot(IS_NOISE,noisy[n-2],'ro',markersize=5)
        else:
            axes[3][2*i+2].plot(NOT_NOISE,clean[1],'bo',markersize=5)
            axes[3][2*i+2].plot(IS_NOISE,noisy[1],'ro',markersize=5)
        if i==1:
            axes[3][2*i+2].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()
    plt.savefig('./res/{}_pi.png'.format(DATA))


def plot_alea():
    fig, axes=plt.subplots(2,2,figsize=(9,6),gridspec_kw={'width_ratios': [3,3]})
    NAME = DATA[6:]
    NAME=NAME.upper()
    plt.suptitle("Dirty {} Aleatoric Uncertainty".format(NAME),fontsize=18)
    for i in range(2):
        t=TYPE[i]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        #axes[0][i].set_title('{} {}\%'.format(noise,RATE),fontsize=18)
        x=[j for j in range(10)]
        x_1=[j+0.2 for j in x]
        #x_2=[i+0.2 for i in x]
        if noise != 'Symmetric':

            if DATA=='dirty_cifar10':
                IS_NOISE=[2,3,4,5,9]
                NOT_NOISE=[0,1,6,7,8]
            elif DATA=='dirty_mnist':
                IS_NOISE = [7,2,5,6,3]
                NOT_NOISE=[0,1,4,8,9]
            else:
                IS_NOISE=[i for i in range(labels)]
                NOT_NOISE=[]
        else:
            IS_NOISE=[j for j in range(10)]
            NOT_NOISE=[]
        axes[0][i].set_title('{} {}\%'.format(noise,RATE),fontsize=16)
        if noise=='Asymmetric':
            axes[0][i].set_ylim((1.0,MAX_SIG+0.003))
        else:    
            axes[0][i].set_ylim((1.0,1.9))
        axes[1][i].set_title('{} {}\%'.format(noise,RATE),fontsize=16)
        x_2=[j-0.2 for j in NOT_NOISE]
        x_3 = [j-0.2 for j in IS_NOISE]
        clean2=np.take(alea_amb[t],NOT_NOISE,axis=0)
        noisy2=np.take(alea_amb[t],IS_NOISE,axis=0)
        MIN_SIG =np.min(alea_clean[t])
        MAX_SIG=np.max(alea_amb[t])
        axes[0][i].set_xticks(x)
        axes[0][i].tick_params(axis='x', labelsize= 12)
        if noise=='Asymmetric':
            axes[0][i].set_ylim((1,0,MAX_SIG+0.003))
        else:    
            axes[0][i].set_ylim((1.0,1.82))
        axes[0][i].set_xlabel("Class",fontsize=14)
        axes[0][i].set_ylabel("Alea Mean",fontsize=14)
        axes[0][i].bar(x,alea_clean[t], width=0.4, align='edge', label='clean instance',color='lightskyblue')
        axes[0][i].plot(x_1,alea_clean[t],'bo',markersize=5)
        #plt.plot(IS_NOISE,noisy,'ro',markersize=5)
        axes[0][i].bar(x,alea_amb[t],width=-0.4, align='edge',label='ambiguous  instance',color='lightpink')
        axes[0][i].plot(x_2,clean2,'bo',markersize=5,label='clean label')
        axes[0][i].plot(x_3,noisy2,'ro',markersize=5,label='noisy label')
        if i==1:    
            axes[0][i].legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.,fontsize='xx-large')
        plt.tight_layout()

        t=TYPE[i+2]
        noise=t[:-4].capitalize()
        RATE= int(float(t[-3:])*100)
        x=[j for j in range(10)]
        x_1=[j+0.2 for j in x]
        #x_2=[i+0.2 for i in x]
        if noise != 'Symmetric':
            if DATA=='dirty_cifar10':
                IS_NOISE=[2,3,4,5,9]
                NOT_NOISE=[0,1,6,7,8]
            elif DATA=='dirty_mnist':
                IS_NOISE = [7,2,5,6,3]
                NOT_NOISE=[0,1,4,8,9]
            else:
                IS_NOISE=[i for i in range(labels)]
                NOT_NOISE=[]
        else:
            IS_NOISE=[j for j in range(10)]
            NOT_NOISE=[]
        if noise != 'Asymmetric' and noise != 'Symmetric':
            if int(noise[-1])==2:
                noise = 'Dual'
            elif int(noise[-1])==3:
                noise = 'Tridiagonal'
        axes[1][i].set_title('{} {}\%'.format(noise,RATE),fontsize=16)
        x_2=[j-0.2 for j in NOT_NOISE]
        x_3 = [j-0.2 for j in IS_NOISE]
        clean2=np.take(alea_amb[t],NOT_NOISE,axis=0)
        noisy2=np.take(alea_amb[t],IS_NOISE,axis=0)
        MIN_SIG =np.min(alea_clean[t])
        MAX_SIG=np.max(alea_amb[t])
        axes[1][i].set_xticks(x)
        axes[1][i].tick_params(axis='x', labelsize= 12)
        if noise=='Asymmetric':
            axes[0][i].set_ylim((1,0,MAX_SIG+0.003))
        else:    
            axes[0][i].set_ylim((1.0,1.82))
        axes[1][i].set_xlabel("Class",fontsize=14)
        axes[1][i].set_ylabel("Alea Mean",fontsize=14)
        axes[1][i].bar(x,alea_clean[t], width=0.4, align='edge', label='clean instance',color='lightskyblue')
        axes[1][i].plot(x_1,alea_clean[t],'bo',markersize=5)
        #plt.plot(IS_NOISE,noisy,'ro',markersize=5)
        axes[1][i].bar(x,alea_amb[t],width=-0.4, align='edge',label='ambiguous  instance',color='lightpink')
        axes[1][i].plot(x_2,clean2,'bo',markersize=5,label='clean label')
        axes[1][i].plot(x_3,noisy2,'ro',markersize=5,label='noisy label')
        #axes[0][i+1].legend(bbox_to_anchor=(-0.1, 1),loc=1, borderaxespad=0.)
        plt.tight_layout()

    plt.savefig('./res/{}_alea.png'.format(DATA))


for i,type in enumerate(TYPE):
    DIR_PATH='./res/rate_res/{}_{}/{}_test.json'.format(DATA,TYPE[i],ID[i])
    with open(DIR_PATH,'r') as json_file:
        data=json.load(json_file)
    transition_clean[type]=data['TM_clean']
    gt_clean[type]=data['GT_clean']
    pi_clean[type]=data['pi_clean']
    dpi_clean[type]=data['dpi_clean']
    transition_amb[type]=data['TM_amb']
    gt_amb[type]=data['GT_amb']
    pi_amb[type]=data['pi_amb']
    dpi_amb[type]=data['dpi_amb']
    alea_amb[type]=data['alea_amb']
    alea_clean[type] = data['alea_clean']
plot_tm()
plot_pi()
plot_alea()
