# Transformer Language Translation

This repository contains two Jupyter notebooks demonstrating the implementation and training of Transformer models for language translation tasks: from German to English and from English to German. These notebooks provide a comprehensive guide to understanding and applying Transformers for sequence-to-sequence tasks, specifically focusing on language translation.

## Theoretical Background

### Transformer Model
The Transformer model, introduced in the paper "Attention is All You Need" by Vaswani et al., represents a significant departure from previous sequence modeling approaches like RNNs and LSTMs. The core idea behind the Transformer is the use of self-attention mechanisms, which compute representations of a sequence by attending to all positions within the same sequence. This allows for parallelization and handling long-range dependencies more effectively.

#### Key components of the Transformer include:
- **Self-Attention Mechanism:** Allows the model to weigh the importance of different words in the input sequence when predicting a word.
- **Positional Encoding:** Since the model does not inherently understand the order of words, positional encodings are added to give the model information about the position of words in the sequence.
- **Multi-Head Attention:** An extension of the attention mechanism that allows the model to focus on different positions, improving the ability to capture information from multiple representation subspaces.
- **Feed-Forward Neural Networks:** Applied to each position separately and identically.
- **Layer Normalization and Residual Connections:** Facilitate training deep networks by stabilizing the activations.

### Sequence-to-Sequence Modeling
Sequence-to-sequence (Seq2Seq) modeling involves converting sequences from one domain (source) to sequences in another domain (target), which is a common setup for tasks like language translation. The Transformer model applies this concept using an encoder-decoder architecture:

- **Encoder:** Processes the input sequence and compresses the information into a context or a set of vectors.
- **Decoder:** Generates the output sequence from the encoded information.

### Training and Evaluation
The notebooks demonstrate the training process, including data preprocessing, model instantiation, loss function definition, and optimizer setup. The training loop involves forward propagation, loss computation, backpropagation, and parameter updates.

Evaluation of the models is performed using the BLEU score, a metric for evaluating the quality of text that has been machine-translated from one natural language to another.

## Notebooks Overview

### German to English Translation
[AnbuValluvan_Devadasan_Assignment9_Transformer_German_to_English.ipynb](AnbuValluvan_Devadasan_Assignment9_Transformer_German_to_English.ipynb) focuses on translating sentences from German to English. It covers data loading, preprocessing, model building, training, and evaluation steps, concluding with translation examples to demonstrate the model's performance.

### English to German Translation
[AnbuValluvan_Devadasan_Assignment9_Translation_English_to_German.ipynb](AnbuValluvan_Devadasan_Assignment9_Translation_English_to_German.ipynb) mirrors the structure of the first notebook but targets the reverse translation task: from English to German. It similarly walks through the entire process of building and training a Transformer model for this task, including examples of translated sentences.

## Getting Started
To run these notebooks:
1. Clone this repository to your local machine or open it in a Jupyter environment.
2. Ensure you have the required dependencies installed, including PyTorch, TorchText, and SpaCy.
3. Open the notebooks and execute the cells sequentially.

## Dependencies
- Python 3.6+
- PyTorch
- TorchText
- SpaCy
- NLTK (for BLEU score evaluation)

## Conclusion
These notebooks provide a practical introduction to using Transformer models for language translation tasks. By following through the notebooks, you will gain insights into the architecture of Transformers, the process of training them, and their application in translating between German and English.

