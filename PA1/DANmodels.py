import torch.nn as nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, read_word_embeddings
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict



class DAN(nn.Module):
    def __init__(self, file: str, input_size: int, hidden_size: int):
        super().__init__()
        self.wordembeddings = read_word_embeddings(file)
        self.embedding_layer = self.wordembeddings.get_initialized_embedding_layer()

        self.embedding_dim = self.embedding_layer.weight.size(1)
        self.layer1 = nn.Linear(self.embedding_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 2)
        self.log_softmax = nn.Softmax(dim=1)

    def forward(self, word_indices):
        size = word_indices.size(0)
        column_tensor = torch.arange(512).repeat(size, 1)
        column_tensor = torch.reshape(column_tensor, (size, 512))

        sentence_embeddings = self.embedding_layer(word_indices)

        mean_vector = sentence_embeddings.mean(dim=1)

        mean_vector = F.relu(self.layer1(mean_vector))
        mean_vector = self.log_softmax(self.layer2(mean_vector))
        print(mean_vector)
        return mean_vector
        # size = word_indices.size(0)
        # column_tensor = torch.arange(512).repeat(size, 1)
        # column_tensor = torch.reshape(column_tensor, (size, 512))

        # sentence_embeddings = self.embedding_layer(word_indices)
        # mean_vector = sentence_embeddings.mean(dim=1)

        # mean_vector = F.relu(self.layer1(mean_vector))
        # mean_vector = self.log_softmax(self.layer2(mean_vector))
        # return mean_vector
    

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, vectorizer=None, train=True):
        # Read the sentiment examples from the input file
        self.examples = read_sentiment_examples(infile)
        # Extract sentences and labels from the examples
        self.sentences = [" ".join(ex.words) for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]
        
        # Vectorize the sentences using CountVectorizer
        if train or vectorizer is None:
            self.vectorizer = CountVectorizer(max_features=512)
            self.embeddings = self.vectorizer.fit_transform(self.sentences).toarray()
        else:
            self.vectorizer = vectorizer
            self.embeddings = self.vectorizer.transform(self.sentences).toarray()
        
        # Convert embeddings and labels to PyTorch tensors
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
    
    def build_vocab(self, sentences):
        vocab = defaultdict(lambda: len(vocab))
        for sentence in sentences:
            for word in sentence.split():
                vocab[word]
        return vocab

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the feature vector and label for the given index
        return self.embeddings[idx], self.labels[idx]




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
    
