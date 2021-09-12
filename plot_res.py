from core.plot_MLN import plot_res
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

TYPE=['symmetric_0.2_','symmetric_0.5_','symmetric_0.8_','asymmetric_0.4_']#,'asymmetric2_0.4_','asymmetric3_0.6_']
ID1=[5]*4 #[11,14,11,11]#[82,85,85,85]#[4,4,4,4]
ID2=[4]*4 #[11,6,1,17]#[26,23,22,24]
MLN_train=[]
MLN_test=[]
MLN_train2=[]
MLN_test2=[]
mode=2
DATA='cifar100'

def plot_res2(train_acc,test_acc,train_acc2,test_acc2,legend,name):
    save_dir='./res/1{}_train_res.png'.format(name)
    plt.figure(figsize=(14,10))
    plt.suptitle('{} Accuracy'.format(name),fontsize=15)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title("{} {} {}".format(name,legend[i][:-5],legend[i][-4:-1]))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(train_acc[i],label='w/ regularizer train',color='k',lw=2,ls='--',marker='')
        plt.plot(test_acc[i],label='w/ regularizer test',color='k',lw=2,ls='-',marker='')
        plt.plot(train_acc2[i],label='w/o regularizer train',color='b',lw=2,ls='--',marker='')
        plt.plot(test_acc2[i],label='w/o regularizer test',color='b',lw=2,ls='-',marker='')
        if i==1:
                plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(save_dir)

def plot_res3(train_acc,test_acc,train_acc2,test_acc2,legend,name):
    save_dir='./res/0{}_train_res.png'.format(name)
    plt.figure(figsize=(14,10))
    plt.suptitle('w/ resamlping {} Accuracy'.format(name),fontsize=15)
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.title("{} {} {}".format(name,legend[i][:-5],legend[i][-4:-1]))
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(train_acc[i],label='baseline train',color='k',lw=2,ls='--',marker='')
        plt.plot(test_acc[i],label='baseline test',color='k',lw=2,ls='-',marker='')
        plt.plot(train_acc2[i],label='MLN train',color='b',lw=2,ls='--',marker='')
        plt.plot(test_acc2[i],label='MLN test',color='b',lw=2,ls='-',marker='')
        if i==1:
                plt.legend(bbox_to_anchor=(1.05, 1),loc=2, borderaxespad=0.)
    plt.tight_layout()
    
    plt.savefig(save_dir)


if mode==1: ## normal
        for i,type in enumerate(TYPE):
                DIR_PATH='./res/log/'+DATA+'_'+type+str(ID1[i])+'.json'
                with open(DIR_PATH,'r') as json_file:
                        data=json.load(json_file)
                MLN_train.append(data['train'])
                MLN_test.append(data['test'])
                Last=data['test'][-5:].copy()
                Last=np.asarray(Last)
                mean=np.mean(Last)
                var=np.sum(abs(Last-mean))/Last.shape[0]
                mean=np.round(mean*100,3)
                var=np.round(var*100,3)
                print('{} ACC: {} +_ {}'.format(type[:-1],mean,var))
        plot_res(MLN_train,MLN_test,TYPE,DATA.upper())
if mode==2: ## Regularizer
        for i,type in enumerate(TYPE):
                DIR_PATH1='./res/log/'+DATA+'_'+type+str(ID1[0])+'.json'
                with open(DIR_PATH1,'r') as json_file:
                        data=json.load(json_file)
                MLN_train.append(data['train'])
                MLN_test.append(data['test'])
                Last=data['test'][190:200].copy()
                Last=np.asarray(Last)
                mean=np.mean(Last)
                var=np.sum(abs(Last-mean**2))/Last.shape[0]
                mean=np.round(mean,3)
                var=np.round(var,3)
                print('{} ACC: {} +_ {}'.format(type[:-1],mean,var))
                DIR_PATH2='./res/log/cifar10_'+type+str(ID1[1])+'.json'
                with open(DIR_PATH2,'r') as json_file:
                        data=json.load(json_file)
                MLN_train2.append(data['train'])
                MLN_test2.append(data['test'])
                Last=data['test'][190:200].copy()
                Last=np.asarray(Last)
                mean=np.mean(Last)
                var=np.sum(abs(Last-mean)**2)/Last.shape[0]
                mean=np.round(mean,3)
                var=np.round(var,3)
                print('{} ACC: {} +_ {}'.format(type[:-1],mean,var))
        plot_res2(MLN_train,MLN_test,MLN_train2,MLN_test2,TYPE,'CIFAR10')

if mode==3: ## trec
        for i,type in enumerate(TYPE):
                DIR_PATH1='./res/log/cnn/mnist_'+type+str(ID1[i])+'.json'
                with open(DIR_PATH1,'r') as json_file:
                        data=json.load(json_file)
                MLN_train.append(data['cnntrain'][:100])
                MLN_test.append(data['cnntest'][:100])
                Last=data['cnntest'][90:100].copy()
                Last=np.asarray(Last)
                mean=np.mean(Last)
                var=np.sum(abs(Last-mean)**2)/Last.shape[0]
                mean=np.round(mean*100,3)
                var=np.round(var*10000,3)
                print('{} ACC: {} +_ {}'.format(type[:-1],mean,var))
                DIR_PATH2='./res/log/trec_'+type+str(ID2[i])+'.json'
                with open(DIR_PATH2,'r') as json_file:
                        data=json.load(json_file)
                MLN_train2.append(data['train'][:100])
                MLN_test2.append(data['test'][:100])
                Last=data['test'][90:100].copy()
                Last=np.asarray(Last)
                mean=np.mean(Last)
                var=np.sum(abs(Last-mean))/Last.shape[0]
                mean=np.round(mean*100,3)
                var=np.round(var*100,3)
                print('{} ACC: {} +_ {}'.format(type[:-1],mean,var))
        plot_res3(MLN_train,MLN_test,MLN_train2,MLN_test2,TYPE,'TREC')