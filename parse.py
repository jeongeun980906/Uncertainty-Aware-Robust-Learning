import os
import argparse
import torch
import random
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# Noise Type
parser.add_argument('--mode', type=str,default='symmetric',help='Noise Pattern')
parser.add_argument('--ER', type=float,default=0.2,help='Noise Rate')
parser.add_argument('--id', type=int,default=1,help='save index')

args = parser.parse_args()

FILE = './res/cifar10_{}_{}/{}_log.txt'.format(args.mode,args.ER,args.id)

mce_log = []
epis_log = []
wd_log = []
with open(FILE,'r') as f:
    i = 0
    while True:
        a = f.readline()
        if i==1:
            lambda1 = a.split('ratio')[1][1:4]
            alpha = a.split('alpha')[1][1:4]
            mixup = a.split('mixup')[1][1]
            k = a.split('k=')[1][0]
            print(lambda1,alpha,mixup)
        if i%4==3:
            if a == '':
                break
            mce = a.split('[')[1].split(']')[0]
            epis = a.split('[')[2].split(']')[0]
            wd = a.split('[')[3].split(']')[0]
            mce_log.append(float(mce))
            epis_log.append(float(epis))
            wd_log.append(float(wd))
        i+=1
plt.title("CIFAR10_{}_{} lambda:{} mixup:{} \nalpha:{} k:{}".format(args.mode,args.ER,lambda1,mixup,alpha,k))
plt.plot(mce_log, label = 'mixture CE loss')
plt.plot(epis_log, label = 'epis')
plt.plot(wd_log,label = 'weight decay loss')
plt.legend()
plt.savefig('./res/cifar10_{}_{}/{}_log.png'.format(args.mode,args.ER,args.id))