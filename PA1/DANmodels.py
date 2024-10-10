import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from sentiment_data import read_word_embeddings

class DAN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # embeddings = self.embedding(word_indices)
        
        # # Compute the average of the embeddings
        # x = torch.mean(embeddings, dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.log_softmax(x)
        return x

# class DAN(nn.Module):
#     def __init__(self, arc: list[int], pretrained_embeddings, vocab_size, embedding_dim):
#         super(DAN, self).__init__()

#         self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
#         if pretrained_embeddings is not None:
#             self.embedding.weight.data.copy_(torch.tensor(pretrained_embeddings))

#         self.layers = []
#         self.nbr_layers = len(arc)
#         for i in range(len(arc)-1):
#             layer = nn.Linear(arc[i], arc[i+1])
#             self.layers.append(layer)
#         self.log_softmax = nn.LogSoftmax(dim=1)

#     def forward(self, word_indices):
#         embeddings = self.embedding(word_indices)
        
#         # Compute the average of the embeddings
#         x = torch.mean(embeddings, dim=1)

#         for layer in self.layers[:-1]:
#             x = F.relu(layer(x))
#         x = self.layers[-1](x)
#         return F.log_softmax(self.layers[self.nbr_layers-1](x))
    
