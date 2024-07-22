## Objective
Implement a simple classifier with a BoW text representation for sentiment analysis.

## SGDClassifier and Bag of Words
SGDClassifier + BoW(1, 1) and then SGDClassifier + BoW(2,2)
Bag of Words and Bag of N-Grams represent a sequence as a count of every word/ngram present in the sequence. This brings up an issue of very sparse and high dimensional vectors. Although SGDClassifier lacks non-linearity and may not provide state-of-the-art results, I used it for its hyperparamter adjustibility and memory flexibility allowing it to train on batches. 

## What is SGDClassifier with log_loss and how does it learn and why did i choose it for Sentiment Analysis
- SGDClassier is a linear model that uses stochastic gradient descent for learning. 
- SGDClassier also allows us to adjust the loss function to use different models like logistic regression, support vector classifier, etc. 
- A method provided by SGDClassifier that helps train the model on a large dataset represented by very sparse vectors is the partial_fit. This is highly suitable in my case due to limited memory and using BoW on 50,000 reviews requires more memory than what is available. In addition, the flexibility of adjusting the parameters of the classifier, like using l1_ratio (regularizer), makes the SGDClassier more suitable and easy to work with. 
- For this SGDClassier I used the 'log_loss' for the loss function and an adaptive learning rate with an initial value of 0.1. 
- Log loss penalizes false classifications. It is sensitive to how confident the classifier is about its predictions. If the true label (yi) is 1 and the model predicts a probability close to 0, the log loss will be very high and thus the weights will change drastically. Conversely, if the model predicts a probability close to 1, the log loss will be low and the weights will slightly be updated. By using log loss as the loss function during training, the model is encouraged to output probabilities that are as close as possible to the actual class labels.
- The combination of log loss with SGD makes the model converge more smoothly, as it adjusts its weights based on the confidence of its predictions.

- The log_loss is represented as: Log Loss=− 1/N ∑ [yi * log(pi) + (1−yi) * log(1−pi)] Where N is the number of samples, yi is the actual label (0 or 1), and pi is the predicted probability of the positive class.

## What is BoW and how is it utilized
- Bag of words counts the presence of words in the sentence and represents the sentence as a vector of the count of words. Although it results in very sparse vectors making it memory and time expensive, it's worth exprimenting with the SGDClassier + BoW and see their sentiment analysis capabilities.

## Process of Training
Data Cleaning: Punctuation, Lem, Stem, Tokenize, Represent as Bag of Words
   - Transforming the reviews into Bag of Words
   - Fitted and Evaluated the model on batches of 1000 reviews
   - Tested the model on 5 manually written inputs