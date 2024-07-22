# What is sentiment analysis
Sentiment analysis uses natural language processing to classify sentences. In this repository, I used the IMDB Dataset, where sentences are classified into two categories: positive and negative.

# Objective of this page
The objective of this project is to build an accurate model that can classify reviews as either positive or negative.

# Summary of the work and results
Finding the best model is not a simple task and requires testing multiple models on the task and choosing the best one. For this reason, I chose to train 2 simple machine learning models and test their performence, then move on to build a more advanced deep learning model and test its performence. The first 2 models I chose were based on the SGDClassifier with its loss function being log_loss, making it a logistic regression model. The first model was trained using a bag of words approach and the second model was trained using an ngram approach of ngrams ranging from 1-2 words. The third model implements the architecture proposed in the paper. This architecture uses RoBERTa's encoder with a GRU and 2 dense layers. The acheived overall accuracies by these models were: 83% accuracy for bag of words model, 85% accuracy for ngram model, and 94% accuracy for RoBERTa GRU model. The RoBERTa GRU model has the highest accuracy and best
fitted algorithm for prediction of positive/negative reviews.

# The Dataset
The IMDB dataset was chosen to train the 3 models. It provides a balanced distribution of 25,000 positive reviews and 25,000 negative reviews. Preprocessing steps were taken to clean the reviews before feeding them to the logistic regression models. These steps included: removal of html tags, removal of trailing white spaces, lowercasing, removal of stopwords, lemmatization, and stemmming. This process was taken in order to remove unnecessary noisy data that might not contribute much to the learning of the models. The reviews were finally tokenized and padded. For the RoBERTa GRU model, no preprecessing was done, since RoBERTa's pretrained tokenizer was used to handle the preprocessing and tokenization of the text. The pretrained tokenizer uses byte-pair level encoding, which is an effective way of reducing vocabulary size while storing meaningful relations between bytes (characters). Data was divided into 80% training and 20% testing.

# SGDClassifier + Bag of Words
The first 2 models were SGDClassifiers. This model's use of Stochastic Gradient Descent makes it an effective learner with an efficient optimization approach on large datasets. In addition, the SGDClassifier allows us to modify its hyperparameters to make it learn like different models. The loss function I used was the log_loss. The log_loss measures how accurate the model is and how confident it is about its prediction. The SGDClassifier was used with an adaptive learning rate that keeps the learning rate equal to eta0, which was initialized as 0.1, as long as the model keeps on improving; otherwise the learning rate becomes eta0 divided by 5. SGDClassifier also provides us with the 'partial_fit' training method which allows the model to train on batches. This method was helpful due to memory limitations. After training the model on batches of 1000 reviews represented as bag of words made of 70,709 words, it acheived an overall accuracy of 83%. The model was also measured on other metrics: 83% precision, 83% recall, 83% ROC AUC, and 82% f1 score.

# SGDClassifier + ngrams
After training the first model on bag of words, another SGDClassifier was trained on 1-2 ngrams. The ngrams approach was used to give the model more context of the next words, in an attempt to improve the model's performence. The second SGClassifier has the exact same hyperparamters as the first: loss='log_loss', learning_rate='adaptive', eta0=0.1. This model was trained on batches of 100 reviews, represented as vectors of 2,641,907 ngrams of 1 or 2 words, using partial_fit. The overall accuracy achieved was 85%. The model was also measured on other metrics: 83% precision, 83% recall, 83% ROC AUC, and 82% f1 score.


# Explain RoBERTa GRU and challenges it solves
In a paper titled "RoBERTa-GRU: A Hybrid Deep Learning Model for Enhanced Sentiment Analysis" by Kian Long Tan, Chin Poo Lee and Kian Ming Lim, a pretrained RoBERTa model was suggested in combination with a GRU layer for sentiment analysis. 

# Explain comparison results with the manual reviews