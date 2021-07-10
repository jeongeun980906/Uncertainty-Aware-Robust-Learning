from typing import Callable

import torch
import torch.utils.data
import torchvision

def make_weights_for_balanced_classes(images, nclasses):                        
    count = [0] * nclasses                                                      
    for item in images:                                                         
        count[item.item()] += 1                                                     
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                 
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])   
    print(weight_per_class)                              
    weight = [0] * len(images)                                              
    for idx, val in enumerate(images):                                          
        weight[idx] = weight_per_class[val]                                  
    return weight    

def new_sampler(train_x,train_y):
    weights = make_weights_for_balanced_classes(train_y, 6)                                                                
    weights = torch.DoubleTensor(weights)                                       
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    return sampler
