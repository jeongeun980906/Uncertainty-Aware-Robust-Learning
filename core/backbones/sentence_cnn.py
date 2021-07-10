import numpy as np
import torch
from torch import nn
import argparse
import time

device='cuda'
class ConvFeatures(nn.Module):
    def __init__(self, word_dimension, filter_lengths, filter_counts, dropout_rate):
        super().__init__()
        conv = []        
        for size, num in zip(filter_lengths, filter_counts): #filter size 별로 초기화
            conv2d = nn.Conv2d(1, num, (size, word_dimension)) # (input_channel, ouput_channel, height, width)
            nn.init.kaiming_normal_(conv2d.weight, mode='fan_out', nonlinearity='relu') # He initialization
            nn.init.zeros_(conv2d.bias)
            conv.append(nn.Sequential(
                conv2d,
                nn.ReLU()
            ))

        self.conv = nn.ModuleList(conv)
        self.filter_sizes = filter_lengths
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, embedded_words):
        features = []
        for filter_size, conv in zip(self.filter_sizes, self.conv): #filter size wise convolution
            # embedded_words: [batch, sentence length, embedding dimension]
            conv_output = conv(embedded_words)
            conv_output = conv_output.squeeze(-1).max(dim=-1)[0]  # max over-time pooling
            features.append(conv_output)
            del conv_output

        features = torch.cat(features, dim=1) # concat each feature from filter
        dropped_features = self.dropout(features)
        return dropped_features


class SentenceCNN(nn.Module):
    def __init__(self, nb_classes, word_embedding_numpy, filter_lengths, filter_counts, dropout_rate):
        super().__init__()

        vocab_size = word_embedding_numpy.shape[0]
        word_dimension = word_embedding_numpy.shape[1]

        # word embedding layer
        self.word_embedding = nn.Embedding(
            vocab_size,
            word_dimension,
            padding_idx=0
        ).to(device)
        # word2vec
        
        self.word_embedding2 = nn.Embedding(
                vocab_size,
                word_dimension,
                padding_idx=0
                ).to(device)
        self.word_embedding.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
        self.word_embedding.weight.requires_grad = False
        self.word_embedding2.weight.detach().copy_(torch.tensor(word_embedding_numpy.astype(np.float32)))
        
        # CNN
        self.features = ConvFeatures(word_dimension, filter_lengths, filter_counts, dropout_rate)

        # fully connected layer
        nb_total_filters = sum(filter_counts)
        self.linear = nn.Linear(nb_total_filters, nb_classes).to(device)
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out', nonlinearity='relu')
        torch.nn.init.zeros_(self.linear.bias)

    def forward(self, input_x):
        x1 = self.word_embedding(input_x).to(device)
        x1 = x1.unsqueeze(1)
        x1 = self.features(x1)
        x2 = self.word_embedding2(input_x).to(device)
        x2 = x2.unsqueeze(1)
        x2 = self.features(x2)
        x=x1+x2
        logits = self.linear(x)
        return logits
    def feature(self, input_x):
        x1 = self.word_embedding(input_x).to(device)
        x1 = x1.unsqueeze(1)
        x1 = self.features(x1)
        x2 = self.word_embedding2(input_x).to(device)
        x2 = x2.unsqueeze(1)
        x2 = self.features(x2)
        x=x1+x2

        return x