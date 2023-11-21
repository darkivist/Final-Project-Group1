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




class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model : int, h: int, dropout:float):
        '''
        :param d_model:  Dimension vector of each token
        :param h: number of heads
        :param dropout: regularization technique, drops nodes randomly.
        '''
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # weights of query
        self.w_k = nn.Linear(d_model, d_model) # weights of key
        self.w_v = nn.Linear(d_model, d_model) # weight of value

        self.w_o = nn.Linear(d_model, d_model) ## Weights of attention layer
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim  = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores


    def forward(self, q, k, v, mask):
        '''
        :param q: Query value
        :param k: Key value
        :param v: Value value
        :param mask: masking tokens we don't want to interact with each other
        :return:
        '''
        query = self.w_q(q)
        key = self.w_q(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        x = x.transpose(1, 2).contigous().view(x.shape[0], -1, self.h * self.d_k)

        return  self.w_o(k)


class ResidualConnection(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    def __int__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.Module([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x ,src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)




class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
        super().__init__()