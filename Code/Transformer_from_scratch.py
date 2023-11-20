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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_lens: int, dropout: float):
        super().__init__()
        self.d_model = d_model
        self.seq_lens = seq_lens
        self.dropout = nn.Dropout(dropout)

        '''
        dropout : Dropout is a regularization technique used to prevent overfitting
        pe : creating positional encoding matrix of seq_lens x d_model
        position : A tensor to give position information or order to the input embeddings. Numerator used for positional encoding calculation
        div_term : denominator used for positional encoding calculation
        pe[:, 0::2] : Sin similarity of the even indices 
        pe[:, 1::2] : cos similarity of the odd indices
        buffer : saves the tensor in the model but doesn't use it as a parameter
        '''
        pe = torch.zeros(seq_lens, d_model)

        position = torch.arange(0, seq_lens, dtype=torch.float).unsqueeze(1)
        position = torch.arange(0, seq_lens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))

        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0) ## Unsqueezing to add extra dimensionn for batch dimension

        self.register_buffer('pe', pe)

    def forward(self,x):
        '''
        requires_grad_(False) : This is so positional encoding values are not used to compute gradients during backpropogation
        self.pe[:, :x.shape[1], :] : This just takes the first 2 dimensions of the tensor - done to match the input layer
        '''

        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):

    def __init__(self, eps: float = 10**-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias  = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean =  x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x-mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout  = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))




class MultiheadAttention(nn.Module):
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()