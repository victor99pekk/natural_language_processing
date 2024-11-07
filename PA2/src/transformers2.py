

import torch.nn as nn
import torch
import torch.nn.functional as F
class Encoder(nn.Model):

    def __init__(self, vocab_size, embd=64):
        self.token_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
        self.postional_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embd)
    
    def forward(self, x:torch.tensor):
        B, T = x.shape
        token_emb = self.token_embeddings(x)
        pos_emb = self.postional_embeddings(x)

        emb_vector = token_emb + pos_emb
        return torch.mean(emb_vector, dim=1)

class Classifier(nn.Module):

    def __init__(self, vocab_size,hidden_size=100, embd=64):
        super().__init__()
        self.encoder = Encoder(vocab_size)
        self.linear = nn.Linear(embd,hidden_size)
        self.linear2 = nn.Linear(hidden_size, 3)
        self.relu = F.ReLu()
        self.softmax = F.softmax()

    def forward(self, x):
        x = self.encoder(x)
        x = self.relu(self.linear(x))
        x = self.softmax(self.linear2(x), dim=1)
        return x