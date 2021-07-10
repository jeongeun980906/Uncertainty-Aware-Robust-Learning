import random
import numpy as np
from numpy.testing import assert_array_almost_equal
from copy import deepcopy
import torch

def rand_bbox(size, lam):
    W = size[1]
    H = size[2]
    cut_rat = np.sqrt(1. - lam)
    cut_rat= np.clip(cut_rat,0.3,0.7)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(int(cut_w/2),int(W-cut_w/2))
    cy = np.random.randint(int(cut_h/2),int(H-cut_h/2))

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def cutmix(data,labels,beta):
    # generate mixed sample
    labels=np.asarray(labels)
    mix_labels=[8,9,6,5,7,3,2,4,0,1]
    for j in range(10):
        indices_a = np.where(labels==j)[0]
        indices_b = np.where(labels==mix_labels[j])[0]
        rand_index = np.random.permutation(indices_b)
        length = indices_a.shape[0]
        shape2 = indices_b.shape[0]
        for i in range(length):
            lam = np.random.beta(beta, beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(data.shape, lam)
            data[indices_a[i], bbx1:bbx2, bby1:bby2, :] = deepcopy(data[rand_index[i%shape2], bbx1:bbx2, bby1:bby2,:])
    return data

def mixup(data,labels,beta):
    # generate mixed sample
    labels=np.asarray(labels)
    mix_labels=[8,9,6,5,7,3,2,4,0,1]
    for j in range(10):
        indices_a = np.where(labels==j)[0]
        indices_b = np.where(labels==mix_labels[j])[0]
        rand_index = np.random.permutation(indices_b)
        length = indices_a.shape[0]
        shape2 = indices_b.shape[0]
        for i in range(length):
            lam = np.random.beta(beta, beta)
            data[indices_a[i]] = lam*data[indices_a[i]]+(1-lam)*data[rand_index[i%shape2]]
    return data