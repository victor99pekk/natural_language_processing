import torch.nn as nn
import torch
import numpy as np

from sentiment_data import read_word_embeddings

class DAN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, pretrained_embeddings_path):

        super(DAN, self).__init__()



    def forward(self, word_indices):
        # Embed the input word indices
        embeddings = self.embedding(word_indices)
        
        # Compute the average of the embeddings
        avg_embeddings = torch.mean(embeddings, dim=1)
        
        # Pass the averaged embeddings through the linear layer
        output = self.linear(avg_embeddings)
        
        return output

    def word_to_index(self, word):
        # Implement this method to map words to indices
        pass
