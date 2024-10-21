import torch.nn as nn
import torch.nn.functional as F
from sentiment_data import read_sentiment_examples, read_word_embeddings
import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict



class DAN(nn.Module):
    def __init__(self, file: str, size1:int=300, size2:int=300, drop_prob:float=0.06):
        super().__init__()
        self.wordembeddings = read_word_embeddings(file)
        self.embedding_layer = self.wordembeddings.get_initialized_embedding_layer()

        self.embedding_dim = self.embedding_layer.weight.size(1)
        self.layer1 = nn.Linear(self.embedding_dim, size1)
        self.layer2 = nn.Linear(size1, size2)
        self.output_layer = nn.Linear(size2, 2)
        self.log_softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, word_indices):

        word_indices = mask_tensor(word_indices)

        sentence_embeddings = self.embedding_layer(word_indices)

        mean_vector = sentence_embeddings.mean(dim=1)
        mean_vector = self.dropout(mean_vector)

        mean_vector = F.relu(self.layer1(mean_vector))
        mean_vector = self.dropout(mean_vector)
        mean_vector = self.dropout(mean_vector)

        mean_vector = F.relu(self.layer2(mean_vector))
        mean_vector = self.log_softmax(self.output_layer(mean_vector))
        return mean_vector
    
def mask_tensor(tensor, mask_prob=0.15):
    """
    Masks values from the input tensor with a given probability.
    :param tensor: Input tensor.
    :param mask_prob: Probability of masking each element.
    :return: Masked tensor.
    """
    # Create a mask with the same shape as the tensor
    mask = (torch.rand(tensor.shape) > mask_prob).int()
    
    # Apply the mask to the tensor
    masked_tensor = tensor * mask
    
    return masked_tensor

class SentimentDatasetDAN(Dataset):
    def __init__(self, infile, word_embeddings, max_len=50):
        """
        Initializes the dataset, converts sentences into word index sequences, and manages labels.
        :param infile: Input file with sentences and labels.
        :param word_embeddings: WordEmbeddings object containing pre-trained embeddings and indexer.
        :param max_len: Maximum sentence length (for padding/truncating).
        """
        self.examples = read_sentiment_examples(infile)
        self.max_len = max_len
        self.word_embeddings = word_embeddings
        self.indexer = word_embeddings.word_indexer

        # Extract sentences and labels
        self.sentences = [ex.words for ex in self.examples]
        self.labels = [ex.label for ex in self.examples]

        # Convert sentences to word index sequences
        self.sentences_idx = [self.convert_sentence_to_indices(sent) for sent in self.sentences]

        # Convert labels to PyTorch tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def convert_sentence_to_indices(self, sentence):
        """
        Converts a sentence (list of words) into a list of word indices using the indexer.
        Unknown words are replaced by the UNK index.
        :param sentence: List of words (a sentence).
        :return: List of word indices, padded/truncated to max_len.
        """
        indices = [self.indexer.index_of(word) if self.indexer.index_of(word) != -1 else self.indexer.index_of("UNK") for word in sentence]
        if len(indices) < self.max_len:
            # Pad sequence
            indices += [self.indexer.index_of("PAD")] * (self.max_len - len(indices))
        else:
            # Truncate sequence
            indices = indices[:self.max_len]
        return indices

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # Return the word indices and corresponding label for the given index
        return torch.tensor(self.sentences_idx[idx], dtype=torch.long), self.labels[idx]