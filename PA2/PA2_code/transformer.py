import torch.nn as nn
import torch
import torch.nn.functional as F

# from torch.nn import functional as F

dropout_prob = 0.15

class Head(nn.Module):
    
    def __init__(self, embd, head_size, dropout=dropout_prob):
        super().__init__()
        self.key = nn.Linear(embd, head_size, bias=False)
        self.query = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        wei = q @ k.transpose(-1, -2) * (32**-0.5)
        wei = F.softmax(wei, dim=1)
        wei = self.dropout(wei)
        out = wei @ v
        return out


class MulitpleHeads(nn.Module):

    def __init__(self, nbr_heads, n_embd, dropout=dropout_prob):
        super().__init__()
        head_size = n_embd // nbr_heads
        self.nbr_heads = nbr_heads
        self.heads = nn.ModuleList([Head(head_size=head_size, embd=n_embd) for _ in range(nbr_heads)])
        self.proj = nn.Linear(head_size * nbr_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=1)
        out = out.transpose(1,2)
        out = self.proj(out)
        return self.dropout(out)
    
class FeedForward(nn.Module):
    def __init__(self, embd):
        super().__init__()
        self.up_layer = nn.Linear(embd, embd*4)
        self.down_layer = nn.Linear(embd*4, embd)
    def forward(self, input):
        input = self.up_layer(input)
        input = F.relu(input)
        return self.down_layer(input)
    
class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
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
        self.heads = MulitpleHeads(nbr_heads=nbr_heads,n_embd=embd)
        self.norm1 = nn.LayerNorm(embd)
        self.norm2 = nn.LayerNorm(embd)
        self.forward_layer = FeedForward(embd)
    def forward(self, input):
        input = input + self.heads(self.norm1(input))
        output = input + self.forward_layer(self.norm2(input))
        return output

class Encoder(nn.Module):
    def __init__(self, vocab_size, embd=64):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
        self.postional_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
        self.block = Block(embd=embd)
        self.norm = LayerNorm(dim=1)
    def forward(self, x:torch.tensor):
        token_emb = self.token_embeddings(x)
        pos_emb = self.postional_embeddings(x)
        emb_vector = token_emb + pos_emb
        emb_vector = self.block(emb_vector)
        mean_vector = torch.mean(emb_vector, dim=1)
        return self.norm(mean_vector)

class Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=100, embd=64, dropout=dropout_prob):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.net = nn.Sequential(
            nn.Linear(embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=1),
            # nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.net(x)
        predicted_class = torch.argmax(x, dim=1)
        return x, predicted_class