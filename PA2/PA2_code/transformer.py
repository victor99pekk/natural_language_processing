import torch.nn as nn
import torch
from torch.nn import Function as F


class Head(nn.Model):
    
    def __init__(self, vocab_size, head_size, block_size):
        super().__init__()
        self.key = nn.Linear(vocab_size, head_size, bias=False)
        self.query = nn.Linear(vocab_size, head_size, bias=False)
        self.value = nn.Linear(vocab_size, head_size, bias=False)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        
        wei = q @ k.transpose(-1, -2) * k.shape[-1]**-0.5
        wei = F.softmax(wei, dim=1)
        out = wei @ v
        return out


class MulitpleHeads(nn.Model):

    def __init__(self, nbr_heads, n_embd):
        head_size = n_embd // nbr_heads
        self.nbr_heads = nbr_heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(nbr_heads)])
        self.proj = nn.Linear(head_size * nbr_heads, n_embd)

    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=1)
        return self.proj(out)
    
class FeedForward(nn.Module):

    def __init__(self, embd):
        self.up_layer = nn.Linear(embd, embd*4)
        self.f = F.ReLu()
        self.down_layer = nn.Linear(embd*4, embd)
    def forward(self, input):
        input = self.up_layer(input)
        input = self.f(input)
        return self.down_layer(input)
    
class LayerNorm(nn.module):

    def __init__(self, dim, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)
    
    def forward(self, x):
        xmean = x.mean(1, keepdim=True) 
        xvar = x.var(1, keepdim=True)
        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
        self.out = self.gamma * xhat + self.beta
        return self.out

    def parameters(self):
        return [self.gamma, self.beta]
    

class Block(nn.Module):

    def __init__(self, embd, nbr_heads=2):
        super().__init__()
        self.heads = MulitpleHeads(nbr_heads, embd)
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.forward_layer = FeedForward(embd)
    
    def forward(self, input):
        input = self.heads(self.norm1(input)) + input
        output = self.forward_layer(self.norm2(input)) + input
        return output

class Encoder(nn.Model):

    def __init__(self, vocab_size, embd=64):
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512)
        self.postional_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=512)
        self.block = Block(embd)
        self.norm = LayerNorm()
    
    def forward(self, x:torch.tensor):
        B, T = x.shape
        token_emb = self.token_embeddings(x)
        pos_emb = self.postional_embeddings(x)

        emb_vector = token_emb + pos_emb
        emb_vector = self.block(emb_vector)
        return torch.mean(emb_vector, dim=-1)
        return self.norm(emb_vector)

# class Classifier(nn.Module):

#     def __init__(self, vocab_size, embd=64):
#         super().__init__()
#         self.encoder = Encoder(vocab_size)
#         self.linear = nn.Linear(embd,3)

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.linear(x)