<h1 align="center">Sentiment Analysis using RoBERTa-GRU and Logistic Regression</h1>

### 🎯 Objective
Sentiment analysis uses natural language processing to classify a sentence according to the emotions expressed in it. In this repository, three models were trained using the IMDb dataset, where sentences are classified into two categories: positive and negative.
The objective of this project is to build, test, and compare 3 sentiment analysis models.

### 📊 Summary of the work and results
As a start, 2 simple machine learning models were trained and tested. Then a more advanced deep learning model was built and tested. The first 2 models were SGDClassifiers using log_loss as their loss function, making them logistic regression models. The first model was [trained using a bag of words approach](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Model%20Training/%5BBoW%5DLogistic_Regression.ipynb) and the second model was [trained using an ngram](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Model%20Training/%5B1_2_grams%5DLogistic_Regression.ipynb) approach of ngrams ranging from 1-2 words. The third model implements the architecture proposed in this [paper](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/PDFs/RoBERTA_GRU_Sentiment_Analysis.pdf). This architecture uses RoBERTa with a GRU and 2 dense layers. The acheived overall accuracies by these models were: 83% accuracy for bag of words model, 85% accuracy for ngram model, and 93.8% accuracy for RoBERTa GRU model. The RoBERTa GRU model has the highest accuracy and best fitted algorithm for prediction of positive and negative reviews.

### 📑 The Dataset
The dataset chosen to train the 3 models was the IMDB dataset. It provides a balanced distribution of 25,000 positive reviews and 25,000 negative reviews. Preprocessing steps were taken to clean the reviews before feeding them to the logistic regression models. These steps included: removal of html tags, removal of trailing white spaces, lowercasing, removal of stopwords, lemmatization, and stemmming. This process was undertaken in order to remove unnecessary noisy data that might not contribute much to the learning of the models. The reviews were finally tokenized and padded. For the RoBERTa GRU model, no preprocessing was done, since RoBERTa's pretrained tokenizer was used to handle the preprocessing and tokenization of the text. The pretrained tokenizer uses byte-pair encoding, which effectively reduces vocabulary size while preserving meaningful relationships between bytes (characters). Data was divided into 80% training and 20% testing.

### ⚡ SGDClassifier and Bag of Words Training
The first model was an SGDClassifier. SGDClassifiers's use of Stochastic Gradient Descent makes it an effective learner with an efficient optimization approach on large datasets. In addition, the SGDClassifier allows us to modify its hyperparameters to make it learn like different models. The loss function I used was the log_loss. The log_loss (negative log-likelihood) measures how accurate the model is and how confident it is about its prediction, the less confident the model is about its prediction the more the weights are penalized. The SGDClassifier was used with an adaptive learning rate that keeps the learning rate equal to eta0, which was initialized as 0.1, as long as the model keeps on improving. Otherwise, the learning rate becomes eta0 divided by 5. SGDClassifier also provides us with the 'partial_fit' training method which allows the model to train on batches. This method was helpful due to memory limitations. After training the model on batches of 1000 reviews represented as bag of words made of 70,709 words, it acheived an overall accuracy of 83%. The model was also measured on other metrics: 83% precision, 83% recall, 83% ROC AUC, and 82% f1 score.<br></br>Training process: [SGDClassifier + Bag of Words](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Model%20Training/%5BBoW%5DLogistic_Regression.ipynb)

### ⚡ SGDClassifier and N-Grams Training
After training the first model on bag of words, another SGDClassifier was trained on 1-2 ngrams. The ngrams approach was used to give the model more context of the next words, in an attempt to improve the model's performence. The second SGClassifier had the exact same hyperparamters as the first: loss='log_loss', learning_rate='adaptive', eta0=0.1. This model was trained on batches of 100 reviews, represented as vectors of 2,641,907 ngrams of 1 or 2 words, using partial_fit. The overall accuracy achieved was 85%. The model was also measured on other metrics: 83% precision, 83% recall, 83% ROC AUC, and 82% f1 score.<br></br>Training process: [SGDClassifier + N-Grams](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Model%20Training/%5B1_2_grams%5DLogistic_Regression.ipynb)

### 👎 Limitations of Bag of Words and N-Grams
The challenges faced with bag of words and ngrams is that they rely on word counts and they fail to capture the contextual relationships between words. For this reason, word embeddings were created to represent words in a vector space capturing the similarity in meaning of the words. In accordance with word embeddings, recurrent neural networks were developed to capture context and meaning in sequential data, like sequences of text. However, RNNs lack long term memory and fail to capture relations between words that are not close to eachother. Gated Recurrent Units (GRU) solve the problem of capturing contextual information in long sequences using gated mechanisms, reset gates and update gates. A hidden state is preserved throughout the GRUs learning process. Reset gates keep the hidden state as it is when the current info is not valuable and update gates update the hidden state when the current info is valuable. Although GRUs are great at learning long-term dependencies in sequences, they are not always capable of capturing enough information from all word embeddings.

### 🌝 RoBERTa GRU
Transformer models are based on attention mechanisms. Attention mechanisms revolutionized the world of NLP by allowing the model to selectively weigh different parts of the input sequence to create informative embeddings resulting in richer word representations. BERT and RoBERTa are biderectional transformer models based on attention mechanisms. They share similar architectures, however they differ in training methods. RoBERTa is an optimized version of BERT (Bidirectional Encoder from Transfomers) and was trained on a much larger dataset. RoBERTa's encoder uses byte pair encoding, self-attention layer and a feed-forward network in order to return word embeddings. Unlike BERT that masks sequences prior to the training, RoBERTa uses dynamic masking where the input sequence is duplicated and different attention masks are applied enabling the RoBERTa model to learn from different input sequences.

### ⚡ RoBERTa GRU Training
In the paper titled ["RoBERTa-GRU: A Hybrid Deep Learning Model for Enhanced Sentiment Analysis" by Kian Long Tan, Chin Poo Lee and Kian Ming Lim](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/PDFs/RoBERTA_GRU_Sentiment_Analysis.pdf), a sentiment analysis model was suggested consisting of a pretrained RoBERTa model with a GRU layer. This architecture leverages the strengths of attention mechanisms and GRUs, providing rich information about the sequences and learning long-range dependendencies in sequences. The paper suggests a pretrained RoBERTa tokenizer for byte-pair encoding of sequences and a RoBERTa model for creating word representations. According to the paper, the RoBERTa GRU model consisted of: a RoBERTa model as an encoder, a GRU layer with 256 hidden states, a Linear layer with 1000 units, GeLu activation function, and Nadam optimizer with a learning rate of 0.00001, and a Linear layer with a softmax activation function. This model achieved a 94.63% accuracy, 95% recall, 95% precision, and 95% f1 score on the IMDB dataset.

For this project, the third and final model used was the RoBERTa GRU architecture with the exact same hyperparameters as proposed in the paper. A RoBERTa pretrained tokenizer for preprocessing, a RoBERTa model as an encoder, a GRU layer with 256 hidden states, a Linear layer with 1000 units, Nadam optimizer, a learning rate of 0.00001, and a GeLu activation function, and a Linear layer with a softmax activation function. After training the model for two epochs and early stopping, with batches of 18, the model achieved 93.8% accuracy, 94.6% precision, 93.3% recall, and 93.9% f1 score.<br></br>Training process: [RoBERTa GRU](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Model%20Training/RoBERTa_GRU.ipynb)

### 🏁 Logistic Regression with BoW and N-Grams 🆚 RoBERTa GRU
In order to further test the 3 models, I wrote a few reviews and tested the models on these reviews.

![Screenshot of reviews](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Screenshots/reviews.png)
![Screenshot of reviews](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Screenshots/LR.png)

![Screenshot of reviews](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/Screenshots/RoBERTa_GRU.png)

We can see that the RoBERTa GRU model correctly classified all the reviews. However, the logistic regression models trained on Bag of Words and NGrams failed to classify some reviews correctly. The logistic regression models seem to struggle with short and simple reviews, unlike the ones they were trained on. Since these models rely solely on word count frequency, they fail to capture the meaning and context of the words. On the other hand, the RoBERTa GRU model is much more complex and uses pretrained embeddings trained on a very large corpus of text. Therefore, it is expected for the RoBERTa GRU model to be accurate after being trained on a downstream task. This also demonstrates the effectiveness of attention mechanisms and transformer models in NLP tasks.

[Full Model Comparison](https://github.com/MarcDagher/RoBERTa_GRU_Sentiment_Analysis/blob/main/test_models.ipynb)
