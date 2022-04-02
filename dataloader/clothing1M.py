################################################################################################
#                                                                                              #
#  code from https://github.com/LiJunnan1992/DivideMix/blob/master/dataloader_clothing1M.py    #
#                                                                                              #
################################################################################################

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image
import json
import torch

import os,sys
import os.path as osp
import json
import codecs
import pickle
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

def mkdir_if_missing(d):
    if not osp.isdir(d):
        os.makedirs(d)

def pickle(data, file_path):
    with open(file_path, 'wb') as f:
        cPickle.dump(data, f, cPickle.HIGHEST_PROTOCOL)

def unpickle(file_path):
    with open(file_path, 'rb') as f:
        data = cPickle.load(f)
    return data

def read_list(file_path, coding=None):
    if coding is None:
        with open(file_path, 'r') as f:
            arr = [line.strip() for line in f.readlines()]
    else:
        with codecs.open(file_path, 'r', coding) as f:
            arr = [line.strip() for line in f.readlines()]
    return arr

def write_list(arr, file_path, coding=None):
    if coding is None:
        arr = ['{}'.format(item) for item in arr]
        with open(file_path, 'w') as f:
            f.write('\n'.join(arr))
    else:
        with codecs.open(file_path, 'w', coding) as f:
            f.write(u'\n'.join(arr))

def read_kv(file_path, coding=None):
    arr = read_list(file_path, coding)
    if len(arr) == 0:
        return [], []
    return zip(*map(str.split, arr))

def write_kv(k, v, file_path, coding=None):
    arr = zip(k, v)
    arr = [' '.join(item) for item in arr]
    write_list(arr, file_path, coding)

def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj

def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))

def compute_matrix_c(clean_labels, noisy_labels):
    cm = confusion_matrix(clean_labels, noisy_labels)
    cm -= np.diag(np.diag(cm))
    cm = cm * 1.0 / cm.sum(axis=1, keepdims=True)
    cm = cm.T
    L = len(cm)
    alpha = 1.0 / (L - 1)
    C = np.zeros((L, L))
    for j in range(L):
        f = cm[:, j].ravel()
        f = zip(f, [i for i in range(L)])
        f = list(f)
        #f.sort(reverse=True)
        sorted(f, reverse=True) # for python3.
        best_lik = -np.inf
        best_i = -1
        for i in range(L + 1):
            c = np.zeros((L,))
            for k in range(0, i):
                c[k] = f[k][0]
            if c.sum() > 0:
                c /= c.sum()
            lik = 0
            for k in range(0, i):
                #lik += f[k][0] * np.log(c[k])
                lik += list(f)[k][0]* np.log(c[k])
            for k in range(i, L):
                #lik += f[k][0] * np.log(alpha)
                lik += list(f)[k][0]* np.log(alpha)
            if lik >= best_lik:
                best_lik = lik
                best_i = i
            if i < L and f[i][0] == 0:
                break
        for k in range(0, best_i):
            C[f[k][1], j] = f[k][0]
    return C / C.sum(axis=0)

def save_to_blobproto(data, output_file):
    shape = (1,) * (4 - data.ndim) + data.shape
    data = data.reshape(shape)
    blob = None
    print("caffe data :: ",data, data.shape)
    num, channels, blob.height, blob.width = data.shape
    blob.data.extend(list(data.ravel().astype(float)))
    with open(output_file, 'wb') as f:
        f.write(blob.SerializeToString())

def make_aux_clean(data_root, output_dir):
    label_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    def _make(token):
        keys = read_list(os.path.join(data_root,
            'clean_{}_key_list.txt'.format(token)))
        lines = [k + ' ' + label_kv[k] for k in keys]
        np.random.shuffle(lines)
        output_file = os.path.join(output_dir, 'clean_{}.txt'.format(token))
        write_list(lines, output_file)
    _make('train')
    _make('val')
    _make('test')

def make_aux_ntype(data_root, output_dir):
    clean_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    noisy_kv = dict(zip(*read_kv(os.path.join(data_root, 'noisy_label_kv.txt'))))
    train_keys = set(read_list(os.path.join(data_root, 'clean_train_key_list.txt')))
    val_keys = set(read_list(os.path.join(data_root, 'clean_val_key_list.txt')))
    test_keys = set(read_list(os.path.join(data_root, 'clean_test_key_list.txt')))
    noisy_keys = set(noisy_kv.keys())
    # compute and save matrix C
    keys = (train_keys | val_keys) & noisy_keys
    clean_labels = np.asarray([int(clean_kv[k]) for k in keys])
    noisy_labels = np.asarray([int(noisy_kv[k]) for k in keys])
    C = compute_matrix_c(clean_labels, noisy_labels)
    save_to_blobproto(C, os.path.join(output_dir, 'matrix_c.binaryproto'))
    # make noise type (ntype)
    def _make(keys, token):
        clean_labels = np.asarray([int(clean_kv[k]) for k in keys])
        noisy_labels = np.asarray([int(noisy_kv[k]) for k in keys])
        lines = []
        alpha = 1.0 / (C.shape[0] - 1)
        for key, y, y_tilde in zip(keys, clean_labels, noisy_labels):
            if y == y_tilde:
                lines.append(key + ' 0')
            elif alpha >= C[y_tilde][y]:
                lines.append(key + ' 1')
            else:
                lines.append(key + ' 2')
        np.random.shuffle(lines)
        output_file = os.path.join(output_dir, 'ntype_{}.txt'.format(token))
        write_list(lines, output_file)
    _make(train_keys & noisy_keys, 'train')
    _make(val_keys & noisy_keys, 'val')
    _make(test_keys & noisy_keys, 'test')

def make_aux_mixed(data_root, output_dir, upsample_ratio=1.0):
    ntype_kv = dict(zip(*read_kv(os.path.join(output_dir, 'ntype_train.txt'))))
    clean_kv = dict(zip(*read_kv(os.path.join(data_root, 'clean_label_kv.txt'))))
    noisy_kv = dict(zip(*read_kv(os.path.join(data_root, 'noisy_label_kv.txt'))))
    clean_keys = read_list(os.path.join(data_root, 'clean_train_key_list.txt'))
    noisy_keys = read_list(os.path.join(data_root, 'noisy_train_key_list.txt'))
    # upsampling clean keys to ratio * #noisy_keys
    clean_keys = np.random.choice(clean_keys, len(noisy_keys) * upsample_ratio)
    # mix clean and noisy data
    keys = list(clean_keys) + list(noisy_keys)
    np.random.shuffle(keys)
    clean, noisy, ntype = [], [], []
    for k in keys:
        if k in clean_kv:
            clean.append(clean_kv[k])
            noisy.append('-1')
        else:
            clean.append('-1')
            noisy.append(noisy_kv[k])
        if k in ntype_kv:
            ntype.append(ntype_kv[k])
        else:
            ntype.append('-1')
    keys = [k + ' -1' for k in keys]
    write_list(keys, os.path.join(output_dir, 'mixed_train_images.txt'))
    write_list(clean, os.path.join(output_dir, 'mixed_train_label_clean.txt'))
    write_list(noisy, os.path.join(output_dir, 'mixed_train_label_noisy.txt'))
    write_list(ntype, os.path.join(output_dir, 'mixed_train_label_ntype.txt'))

class clothing1M(Dataset): 
    def __init__(self, root, transform, mode, num_samples=0, pred=[], probability=[], paths=[], num_class=14): 
        
        self.root = root
        self.transform = transform
        self.mode = mode
        self.train_labels = {}
        self.train_labels_map = {}
        self.test_labels_map = {}
        self.test_labels = {}
        self.val_labels = {}            
        self.category_names_eng = []
        self.transition_matrix = None

        with open('%s/category_names_eng.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            self.category_names_eng = lines
        
        with open('%s/noisy_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()     
                fnum = entry[0][7:].split("/")[0]
                tmp_name = entry[0][7:].split("/")[-1].split(',')
                img_name = tmp_name[0] + "_" + tmp_name[-1]
                img_path = self.root + "/noisy_train/" + entry[1] + "/" +img_name 
                img_path_f = self.root + "/noisy_train/" + fnum + "/" +img_name 
                self.train_labels[img_path] = int(entry[1])
                self.train_labels_map[img_path_f] = img_path

        with open('%s/clean_label_kv.txt'%self.root,'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()           
                fnum = entry[0][7:].split("/")[0]
                tmp_name = entry[0][7:].split("/")[-1].split(',')
                img_name = tmp_name[0] + "_" + tmp_name[-1]
                img_path = self.root + "/clean_test/" + entry[1] + "/" +img_name 
                img_path_f = self.root + "/clean_test/" + fnum + "/" +img_name 
                self.test_labels[img_path] = int(entry[1])   
                self.test_labels_map[img_path_f] = img_path

        if mode == 'train':
            train_imgs=[]
            with open('%s/noisy_train_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    fnum = l.split("/")[1]
                    tmp_name = l.split("/")[-1].split(',')
                    img_name = tmp_name[0] +"_" + tmp_name[-1]
                    img_path = self.root + "/noisy_train/" + fnum + "/" +img_name 
                    train_imgs.append(self.train_labels_map[img_path])                                
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            for impath in train_imgs:
                label = self.train_labels[impath] 
                if class_num[label]<(num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append(impath)
                    class_num[label]+=1
            random.shuffle(self.train_imgs)     
        elif mode=='test':
            self.test_imgs = []
            with open('%s/clean_test_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    fnum = l.split("/")[1]
                    tmp_name = l.split("/")[-1].split(',')
                    img_name = tmp_name[0] +"_" + tmp_name[-1]
                    img_path = self.root + "/clean_test/" + fnum + "/" +img_name 
                    self.test_imgs.append(self.test_labels_map[img_path])     
        
        elif mode=='val':
            self.val_imgs = []
            with open('%s/clean_val_key_list.txt'%self.root,'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    fnum = l.split("/")[1]
                    tmp_name = l.split("/")[-1].split(',')
                    img_name = tmp_name[0] +"_" + tmp_name[-1]
                    img_path = self.root + "/clean_test/" + fnum + "/" +img_name 
                    self.val_imgs.append(self.test_labels_map[img_path])   
                    
    def __getitem__(self, index):  
        if self.mode=='train':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image)
            return img, target   
        elif self.mode=='test':
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target
        elif self.mode=='val':
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]     
            image = Image.open(img_path).convert('RGB')   
            img = self.transform(image) 
            return img, target    
        
    def __len__(self):
        if self.mode=='test':
            return len(self.test_imgs)
        if self.mode=='val':
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)            