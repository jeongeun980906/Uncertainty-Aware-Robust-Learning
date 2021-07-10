import torch
import numpy as np
import time
import random
import re
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
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
    def load_trec(self):
        x_text, y = dh.load_trec_data("./data/TREC/traindata.txt")
        word_id_dict, _ = dh.buildVocab(x_text, 30000)  # training corpus를 토대로 단어사전 구축
        self.vocab_size = len(word_id_dict) + 4
        for word in word_id_dict.keys():
            word_id_dict[word] += 4  # <pad>: 0, <unk>: 1, <s>: 2 (a: 0 -> 4)
        word_id_dict['<pad>'] = 0  # zero padding을 위한 토큰
        word_id_dict['<unk>'] = 1  # OOV word를 위한 토큰
        word_id_dict['<s>'] = 2  # 문장 시작을 알리는 start 토큰
        word_id_dict['</s>'] = 3  # 문장 마침을 알리는 end 토큰
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
        train_y=np.asarray(train_y)
        temp=np.where(train_y==5)
        print(temp[0].shape[0])
        temp=np.where(train_y==4)
        print(temp[0].shape[0])
        temp=np.where(train_y==3)
        print(temp[0].shape[0])
        temp=np.where(train_y==2)
        print(temp[0].shape[0])
        temp=np.where(train_y==1)
        print(temp[0].shape[0])
        temp=np.where(train_y==0)
        print(temp[0].shape[0])
        if self.noise_type=='symmetric':
            train_y,trainsition_matrix=noisify_trec_symmetric(train_y,self.noise_rate,nb_classes=6)
        elif self.noise_type=='asymmetric':
            train_y,trainsition_matrix=noisify_trec_asymmetric(train_y,self.noise_rate)
        else:
            trainsition_matrix=np.eye(6)
        temp=np.where(train_y==5)
        print(temp[0].shape[0])
        temp=np.where(train_y==4)
        print(temp[0].shape[0])
        temp=np.where(train_y==3)
        print(temp[0].shape[0])
        temp=np.where(train_y==2)
        print(temp[0].shape[0])
        temp=np.where(train_y==1)
        print(temp[0].shape[0])
        temp=np.where(train_y==0)
        print(temp[0].shape[0])
        train_y = torch.tensor(train_y,dtype=torch.int64)

        test_x = dh.sequence_to_tensor(test_x, nb_paddings=(nb_pad, nb_pad))
        val_y = torch.tensor(test_y,dtype=torch.int64)
        test_y=np.asarray(test_y)
        temp=np.where(test_y==5)
        if self.noise_type=='symmetric':
            test_y,_=noisify_trec_symmetric(test_y,self.noise_rate,nb_classes=6)
        elif self.noise_type=='asymmetric':
            test_y,_=noisify_trec_asymmetric(test_y,self.noise_rate)
        test_y = torch.tensor(test_y,dtype=torch.int64)
        self.len_test=test_y.size(0)
        self.len_train=train_y.size(0)
        if self.use_sampler:
            print('use sampler')
            sampler=new_sampler(train_x,train_y)                     
            training_loader = DataLoader(TensorDataset(train_x, train_y),sampler=sampler, batch_size=self.batch_size, num_workers=1)
        else:
            training_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=self.batch_size, shuffle=True, num_workers=1)
        val_loader = DataLoader(TensorDataset(test_x, val_y), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=self.batch_size, shuffle=False)
        
        return training_loader,val_loader,test_loader,trainsition_matrix

    def embedding(self,embedding_size=300,word2vec = "./data/GoogleNews-vectors-negative300.bin"):
        print("Loading W2V data...")
        pre_emb = KeyedVectors.load_word2vec_format(word2vec, binary=True)  # pre-trained word2vec load
        pre_emb.init_sims(replace=True)
        # num_keys = len(pre_emb.vocab)
        # print("loaded word2vec len ", num_keys)

        # initial matrix with random uniform, pretrained word2vec으로 vocabulary 내 단어들을 초기화하기 위핸 weight matrix 초기화
        initW = np.zeros((self.vocab_size, embedding_size)) # embedding 300
        # load any vectors from the word2vec
        print("init initW cnn.W in FLAG")
        for w in self.word_id_dict.keys():
            arr = []
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in pre_emb:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                arr = pre_emb[w]  # word2vec vector를 가져옴
            elif w.lower() in pre_emb:  # 소문자로도 확인
                arr = pre_emb[w.lower()]
            elif s in pre_emb:  # 전처리 후 확인
                arr = pre_emb[s]
            elif s.isdigit():  # 숫자이면
                arr = pre_emb['1']
            if len(arr) > 0:  # 직접 구축한 vocab 내 단어가 google word2vec에 존재하면
                idx = self.word_id_dict[w]  # 단어 index
                initW[idx] = np.asarray(arr).astype(np.float32)  # 적절한 index에 word2vec word 할당
            initW[0] = np.zeros(embedding_size)
        print(initW)
        return initW