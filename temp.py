import json
import numpy as np

DATA = 'cifar10_'#'mnist_'
noise = ['instance_0.2','instance_0.4']
#['symmetric_0.2','symmetric_0.5','symmetric_0.8','asymmetric_0.4']  
 
ID = 13
for n in noise:
    DIR_PATH='./res/'+DATA+n+'/'+str(ID)+'_log.json'
    try:
        with open(DIR_PATH,'r') as json_file:
            data=json.load(json_file)
            res = data['test']
        last = res[-5:]
        last = np.asarray(last)*100
        mean = np.mean(last)
        std = np.sum(np.abs(last-mean),axis=0)/last.shape[0]
        print(n,round(mean,2),'$\pm$',round(std,2))
    except:
        print('no file')
        pass