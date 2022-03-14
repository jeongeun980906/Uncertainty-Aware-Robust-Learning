import json
import numpy as np

DATA = 'dirty_cifar10_asymmetric_0.4'
ID = 1
DIR_PATH='./res/'+DATA+'/'+str(ID)+'_log.json'
with open(DIR_PATH,'r') as json_file:
    data=json.load(json_file)
    res = data['test']

last = res[-10:]
last = np.asarray(last)*100
mean = np.mean(last)
std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
print(round(mean,2),'$\pm$',round(std,2))