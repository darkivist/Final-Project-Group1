import torch
import torch.nn as nn
import math

'''
Hi, A lot of comments here would be just me talking to myself and reminding what the function is for and is doing.
'''

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):  ## This is called a constructor. Initializes the instance variables or you could say the inputs
        '''
        vocab size : as the name suggests it's the unique words in our data
        d_model : Dimension vector of each word/token
        '''
        super().__init__()  ## super() is used to call a method from the parent class ( in this case the parent class is Neural Network for pytorch)
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding  = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        ## Using nn.Embedding to create an input layer
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, seq_lens: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_lens = seq_lens
        self.dropout = nn.Dropout(dropout)

