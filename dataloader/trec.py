import torch
import numpy as np
import time
import random
import re
from torch.utils.data import TensorDataset, DataLoader
import dataloader.preprocessing.data_helper as dh
from dataloader.utils import noisify_trec_asymmetric,noisify_trec_symmetric
from gensim.models.keyedvectors import KeyedVectors
from dataloader.preprocessing.resampling import new_sampler

class TREC():
    def __init__(self,noise_type=None,noise_rate=0.2,batch_size=128,test_sample_percentage=0.1,filter_list = [3,4,5],IB=False):
        self.noise_type=noise_type
        self.noise_rate=noise_rate
        self.batch_size=batch_size
        self.test_sample_percentage=test_sample_percentage
        self.filter_lengths= filter_list
        self.use_sampler=IB
    def load_trec(self,cv,indicies=None):
        x_text, y = dh.load_trec_data("./data/TREC/traindata.txt")
        word_id_dict, _ = dh.buildVocab(x_text, 30000)  # training corpus
        self.vocab_size = len(word_id_dict) + 4
        for word in word_id_dict.keys():
            word_id_dict[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
        word_id_dict['<pad>'] = 0  # zero padding token
        word_id_dict['<unk>'] = 1  # OOV word token
        word_id_dict['<s>'] = 2  # start token
        word_id_dict['</s>'] = 3  # end token
        self.word_id_dict=word_id_dict
        x_indices = dh.text_to_indices(x_text, word_id_dict, True)
        data = list(zip(x_indices, y))
        random.seed(0)
        random.shuffle(data)
        x_indices, y = zip(*data)

        test_sample_index = -1 * int(self.test_sample_percentage * float(len(y)))
        train_x, test_x = x_indices[:test_sample_index], x_indices[test_sample_index:]
        train_y, test_y = y[:test_sample_index], y[test_sample_index:]
        nb_pad = int(max(self.filter_lengths) / 2 + 0.5)
        train_x = dh.sequence_to_tensor(train_x, nb_paddings=(nb_pad, nb_pad))
        train_clean_y=np.asarray(train_y)

        if self.noise_type=='symmetric':
            train_y,trainsition_matrix=noisify_trec_symmetric(train_clean_y,self.noise_rate,nb_classes=6)
        elif self.noise_type=='asymmetric':
            train_y,trainsition_matrix=noisify_trec_asymmetric(train_clean_y,self.noise_rate)
        else:
            trainsition_matrix=np.eye(6)
        noise_or_not = np.transpose(train_y)==np.transpose(train_clean_y)
        train_y = torch.tensor(train_y,dtype=torch.int64)

        test_x = dh.sequence_to_tensor(test_x, nb_paddings=(nb_pad, nb_pad))
        val_x = test_x
        val_y = torch.tensor(test_y,dtype=torch.int64)
        test_y=np.asarray(test_y)

        if self.noise_type=='symmetric':
            test_y,_=noisify_trec_symmetric(test_y,self.noise_rate,nb_classes=6)
        elif self.noise_type=='asymmetric':
            test_y,_=noisify_trec_asymmetric(test_y,self.noise_rate)
        test_y = torch.tensor(test_y,dtype=torch.int64)
        self.len_test=test_y.size(0)
        self.len_train=train_y.size(0)
        if indicies != None:
            indicies = torch.tensor(indicies,dtype=torch.int64)
            val_x = train_x[indicies]
            val_y = train_y[indicies]
            actual_noise_rate = noise_or_not[indicies].sum()/(indicies.size(0))
        else:
            actual_noise_rate = noise_or_not.sum()/(noise_or_not.shape[0])
        if self.use_sampler:
            print('use sampler')
            sampler=new_sampler(train_x,train_y)       
            if cv:               
                training_loader = DataLoader(TensorDataset(train_x, train_y),sampler=sampler, batch_size=self.batch_size, num_workers=1,shuffle=False)
            else:
                training_loader = DataLoader(TensorDataset(train_x, train_y),sampler=sampler, batch_size=self.batch_size, num_workers=1,shuffle=True)
        else:
            training_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        
        return training_loader,val_loader,test_loader,trainsition_matrix,actual_noise_rate

    def embedding(self,embedding_size=300,word2vec = "./data/GoogleNews-vectors-negative300.bin"):
        print("Loading W2V data...")
        pre_emb = KeyedVectors.load_word2vec_format(word2vec, binary=True)  # pre-trained word2vec load
        pre_emb.init_sims(replace=True)
        # num_keys = len(pre_emb.vocab)
        # print("loaded word2vec len ", num_keys)

        # initial matrix with random uniform, pretrained word2vec
        initW = np.zeros((self.vocab_size, embedding_size)) # embedding 300
        # load any vectors from the word2vec
        print("init initW cnn.W in FLAG")
        for w in self.word_id_dict.keys():
            arr = []
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in pre_emb:  
                arr = pre_emb[w]  
            elif w.lower() in pre_emb:  
                arr = pre_emb[w.lower()]
            elif s in pre_emb:  
                arr = pre_emb[s]
            elif s.isdigit(): 
                arr = pre_emb['1']
            if len(arr) > 0:  
                idx = self.word_id_dict[w] 
                initW[idx] = np.asarray(arr).astype(np.float32) 
            initW[0] = np.zeros(embedding_size)
        return initW