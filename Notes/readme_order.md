# What is sentiment analysis
Sentiment analysis uses natural language processing to classify sentences. In this repository, I used the IMDB Dataset, where sentences are classified into two categories: positive and negative.

# Objective of this page
The objective of this project is to build an accurate model that can classify reviews as either positive or negative.

# Summary of what i did and outline of how i will present the info
Finding the best model is not a simple task and requires testing multiple models on the task and choosing the best one. For this reason, I chose to train 2 simple machine learning models and test their performence, then move on to build a more advanced deep learning model and test its performence. The first 2 models I chose were based on the SGDClassifier with its loss function being log_loss, making it a logistic regression model. The first model was trained using a bag of words approach and the second model was trained using an ngram approach of ngrams ranging from 1-2 words. The third model implements the architecture proposed in the paper. This architecture uses RoBERTa's encoder with a GRU and 2 dense layers. The acheived overal accuracies by these models were: 83% accuracy for bag of words model, 

The dataset I chose is the IMDB Dataset. After the data preprocessing phase, I trained 3 models, evaluated them using the accuracy score, and tested them on a few manually written reviews. I, I will 

# Explain the use of SGDCLassifier with BoW then ngrams + Challenges faced that led me to use SGDClassifier

# Explain RoBERTa GRU and challenges it solves

# Explain comparison results with the manual reviews