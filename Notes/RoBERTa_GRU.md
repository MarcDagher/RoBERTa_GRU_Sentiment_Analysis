<h1>RoBERTa GRU</h1>

## Objective
- The goal of this notebook is to try to implement the RoBERTa-GRU architecture on the IMDB Dataset for sentiment analysis and replicate the results of this paper. 
- This paper suggests a RoBERTa-GRU model for sentiment analysis, leveraging the strengths of transformer model and recurrent neural network.

## The way the model works: 
-- The encoder part of RoBERTa is used to tokenize the data and return word embeddings. The word embeddings are passed on to the GRU to capture long-range dependencies. A dense layer then learns the relationships between the GRU output and the class labels. Softmax layer outputs class probabilities.

## Why is this architecture effective?
- RoBERTa uses byte-level byte pair encoding for tokenization resulting in a smaller vocab and making it compuationally more effective.
- The RoBERTa encoder relies on Attention Mechanism that weighs different parts of the input sequence to create informative embeddings.
- RoBERTa uses dynamic masking where the input sequence is duplicated and different attention masks are applied enabling the RoBERTa model to learn from different input sequences.
- RoBERTa was trained on a much larger dataset making it capable of learning more relations.
- GRU finally captures long-term dependencies using gated mechanisms which are computationally efficient.

## Paper's training:
- Hyperparameters:
-- Hidden states in the GRU: 256
-- Nadam optimizer 
-- Learning rate: 0.00001
- Data: 60% training data - 20 % validation - 20% testing 
- Max 100 epochs 
- Batch size 32

## My Training:
- Hyperparameters: same as paper
- Data: 80% training 20% testing
- Batch size: 18

# What is RoBERTa
RoBERTa is an optimized version of BERT (Bidirectional Encoder from Transfomers) trained on a much larger dataset. The encoder uses byte pair encoding, self-attention layer and a feed-forward network in order to return return word embeddings. RoBERTa uses dynamic masking where the input sequence is duplicated and different attention masks are applied enabling the RoBERTa model to learn from different input sequences.

# What is GRU
GRU is an optimized RNN. It relies on gated mechanisms to capture long-term context in the sequence.

# What is transformer and attention
Attention is an alternative to embeddings. It adds the position of the word it's context vector, emphasing the importance of the word in the sequence.

# Why both
For this sentiment analysis task, combining RoBERTa's advanced process of learning the context of the sequence using a sequence to sequence architecture with attention mechanism and GRU's ability to capture long-term relationships through memory cells and gated units, serves as a powerful model combining the strengths of attention and long term memory.
