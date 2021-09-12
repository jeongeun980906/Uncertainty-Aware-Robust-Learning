import matplotlib.pyplot as plt
import numpy as np
import argparse,json

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

DATA='mnist' #'mnist'
mode='symmetric'
ER=[(i+1)/10 for i in range(8)]
#ID=[3 for _ in range(8)]
#EPOCH=[116,199,58]
pi_entropy=[]
for n,rate in enumerate(ER):
    DIR='./res/rate_res/{}_{}_{}/{}_test.json'.format(DATA,mode,rate,6)
    with open(DIR,'r') as json_file:
        data=json.load(json_file)
    pi=data['dpi']
    pi=np.asarray(pi)
    pi = pi[:,0]
    pi_avg=np.average(pi)
    pi_entropy.append(pi_avg)
pi_entropy.reverse()
plt.figure()
plt.title('Noise Rate Estimation',fontsize=20)
plt.bar(ER,pi_entropy,alpha=0.5,width=0.08)
plt.plot(ER,pi_entropy,color='k',marker='o')
plt.xlabel('noise rate',fontsize=20)
plt.ylabel('$\pi (1) - \pi (2)$',fontsize=15)
plt.xticks(fontsize=13)
plt.savefig('./res/{}_rate.png'.format(DATA))

