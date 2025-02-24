# Transformer Blocks

## Overview
This project implements different components of the Transformer architecture from scratch. The project consists of two main parts:

1. **Transformer Encoder with Classifier**: A Transformer encoder is trained alongside a feedforward classifier to predict which politician delivered a given speech segment.
2. **GPT-like Transformer Decoder**: A word-level decoder is pretrained on an autoregressive language modeling task, reporting perplexity on different politicians' speeches.

## Getting Started
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- NumPy
- NLTK
- Matplotlib (for visualization)

You can install the required libraries using:
```bash
pip install torch numpy nltk matplotlib````markdown
# Transformer Blocks - CSE156 PA2

## Overview
This project implements different components of the Transformer architecture from scratch as part of the CSE156 PA2 assignment. The project consists of two main parts:

1. **Transformer Encoder with Classifier**: A Transformer encoder is trained alongside a feedforward classifier to predict which politician delivered a given speech segment.
2. **GPT-like Transformer Decoder**: A word-level decoder is pretrained on an autoregressive language modeling task, reporting perplexity on different politicians' speeches.

## Getting Started
### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.x
- PyTorch
- NumPy
- NLTK
- Matplotlib (for visualization)

You can install the required libraries using:
```bash
pip install torch numpy nltk matplotlib
````

### make

```bash
python main.py --part1
```

Expected accuracy: \~80% on test set.

### Part 2: Transformer Decoder for Language Modeling

Pretrain and evaluate the decoder:

```bash
python main.py --part2
```

Expected perplexity:

- Training set: \~100s
- Test sets: \~300-400s

## Evaluation and Reporting

- **Attention Visualization**: Use `utilities.py` to validate attention matrices.
- **Performance Metrics**:
  - Accuracy for classification (reported after each epoch up to 15 epochs).
  - Perplexity for language modeling (reported after every 100 iterations up to 500 iterations).
- **Model Parameters**: Number of parameters for both encoder and decoder should be included in the report.

## Submission

Submit the following files on Gradescope:

- **Code**: Ensure all scripts are included.
- **README**: This file, containing execution instructions.
- **Report**: A 5-page PDF summarizing implementation, results, and observations.

## References

- Vaswani et al. (2017), *Attention is All You Need*.
- Andrej Karpathy’s Transformer tutorial: [YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)

---

**Author:** Your Name

```
```
