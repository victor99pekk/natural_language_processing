import torch.nn as nn
import torch
import torch.nn.functional as F

# from torch.nn import functional as F

dropout_prob = 0.1
block_size = 32  # Maximum context length for predictions
vocab_size = 5755

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

class Head(nn.Module):
    
    def __init__(self, embd, head_size, dropout=dropout_prob):
        super().__init__()
        self.key = nn.Linear(embd, head_size, bias=False)
        self.query = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.att_mat = None
        # self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, input):
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        wei = q @ k.transpose(-1, -2) * (32**-0.5)
        wei = F.softmax(wei, dim=2)
        self.att_mat = wei
        wei = self.dropout(wei)
        out = wei @ v
        return out

class MaskedHead(nn.Module):
    def __init__(self, embd, head_size, dropout=dropout_prob) -> None:
        super().__init__()
        self.key = nn.Linear(embd, head_size, bias=False)
        self.query = nn.Linear(embd, head_size, bias=False)
        self.value = nn.Linear(embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size, dtype=float)))
        self.att_mat = None
    def forward(self, input):
        B,T,C = input.shape
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)
        wei = q @ k.transpose(-1, -2) *  (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T][:T]==0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        self.att_mat = wei
        wei = self.dropout(wei)
        out = wei @ v
        return out
        


class MulitpleHeads(nn.Module):
    def __init__(self, nbr_heads, n_embd, dropout=dropout_prob, mask_future_tokens:bool=True):
        super().__init__()
        head_size = n_embd // nbr_heads
        self.nbr_heads = nbr_heads
        if not mask_future_tokens:
            self.heads = nn.ModuleList([Head(head_size=head_size, embd=n_embd) for _ in range(nbr_heads)])
        else:
            self.heads = nn.ModuleList([MaskedHead(head_size=head_size, embd=n_embd) for _ in range(nbr_heads)])
        self.proj = nn.Linear(head_size * nbr_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    def forward(self, input):
        out = torch.cat([head(input) for head in self.heads], dim=1)
        out = out.transpose(1,2)
        out = self.proj(out)
        return self.dropout(out)
    def list_of_attentions(self):
        return [head.att_mat for head in self.heads]
    
class FeedForward(nn.Module):
    def __init__(self, embd):
        super().__init__()
        self.up_layer = nn.Linear(embd, embd*4)
        self.down_layer = nn.Linear(embd*4, embd)
    def forward(self, input):
        input = self.up_layer(input)
        input = F.relu(input)
        return self.down_layer(input)

class Block(nn.Module):
    def __init__(self, embd, nbr_heads=2, mask_future_tokens=True, hidden_size=None):
        super().__init__()
        hidden_size = embd*4 if hidden_size is None else hidden_size
        self.heads = MulitpleHeads(nbr_heads=nbr_heads,n_embd=embd,mask_future_tokens=mask_future_tokens)
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
        self.block = Block(embd=embd, mask_future_tokens=False)
        self.norm = LayerNorm(dim=1)
    def forward(self, x:torch.tensor):
        token_emb = self.token_embeddings(x)
        pos_emb = self.postional_embeddings(x)
        emb_vector = token_emb + pos_emb
        emb_vector = self.block(emb_vector)
        mean_vector = torch.mean(emb_vector, dim=1)
        return self.norm(mean_vector), self.block.heads.list_of_attentions()

class Decoder(nn.Module):
    def __init__(self,embd, vocab_size, hidden_size=100):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
        self.postional_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
        self.block = Block(embd=embd)
        self.norm = LayerNorm(dim=1)
        self.feedForward_layer = nn.Sequential(
            nn.Linear(embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, vocab_size),
            nn.Dropout(0.35)
        )
    
    def forward(self, x, y=None):
        # y = F.one_hot(y, num_classes=vocab_size).float()
        # print(y)
        token_emb = self.token_embeddings(x)    # B, T, C
        pos_emb = self.postional_embeddings(x)
        emb_vector = token_emb + pos_emb
        emb_vector = self.block(emb_vector)
        B,T,C = emb_vector.shape
        emb_vector = self.norm(emb_vector) # B,T,C
        feedforward_output = self.feedForward_layer(emb_vector) # B,T, vocab_size
        feedforward_output = feedforward_output.view(B*T, vocab_size)
        if y == None:
            return feedforward_output, self.block.heads.list_of_attentions()
        y = y.view(B*T)
        return F.cross_entropy(feedforward_output, y), self.block.heads.list_of_attentions()
    

class Classifier(nn.Module):
    def __init__(self, vocab_size, hidden_size=100, embd=64):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.net = nn.Sequential(
            nn.Linear(embd, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 3),
            nn.Dropout(0.1),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        x = self.net(x)
        predicted_class = torch.argmax(x, dim=1)
        return x, predicted_class