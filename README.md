# SMS-Spam-Detection-And-Sentimental-LIAR-Detection-Using-Naive-Bayes

In this project, we will try to do classification on two datasets using Naïve Bayes. First, we separate spam and non-spam SMS on one dataset, and then we train a classification tool to detect lies and truth in another dataset. The purpose of doing this exercise is to get more familiar with this type of classification and practice extracting suitable features.

### Datasets

* https://www.kaggle.com/uciml/sms-spam-collection-dataset
* https://github.com/UNHSAILLab/SentimentalLIAR

\* for running the code make sure you have these files near the two notebook files:
* Dataset #1: spam.csv
* Dataset #2: train_final.csv, test_final.csv
* Spam_words.txt, intensifiers.txt

## Part 1: Spam SMS detection (Applying Naïve Bayes on the first dataset)
### Preprocess:
  * **Tokenizing** by `word_tokenize()` in nltk library.
  * **Normalizing** and convert all letters to lower case by `lower()`. This is important because we are using **BOW** model and naïve bayes and two words “Congratulations” and “congratulations” are now different so, this can hep us to detect same words better and therefore classify better.
  * **Removing stop words** like prepositions detecting by `nltk_stopwords()`. Stop words are words like prepositions which has a high frequency in all classes to be detected so they are frequent in all classes and do not help us to classify hence, we remove them.
  * **Lemmatizing** tokens by `WordNetLemmatizer()` in `nltk`.
  
### Feature Extraction:
Feature engineering is one of the most important steps in machine learning. It is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If we can use suitable features and feed them to our model then the model will be able to understand the sentence better. For detecting spam and ham in this dataset I extracted thes features (based on my experiments and [this website](https://www.researchgate.net/figure/List-of-extracted-features-for-spam-and-ham-messages_tbl2_340607093)):
  * **Frequency of words**: we use bag of words model. We use the assumption of independency and count the number of words in a record.
  * **Message length**: mostly spam messages are longer than non-spam messages. So, this length could be an important feature to extract.
  * **Number of uppercased words**: mostly spam messages have uppercased words to get your attention and make you see some keywords such as “FREE” to get curious and read the whole message.
  * **URL presence**: usually spam messages contain a URL to invite receivers to checkout that link. In non-spam messages this happens rarely.
  * **Phone number presence**: normally we see a phone number in a spam message to invite the user to call to get more information.
  * **Currency sign presence**: in a spam message we usually see a price declaration and it is mostly followed by a currency sign like “$”
  * **Number of spam keywords**: there are some keywords like “congratulations” or “free” which mainly occur in spam massages. I prepared a list of these words in a “spam_words.txt”.



  
### Training the model:
I employed `MultinomialNB()` in `sklearn` library to train naïve bayes classifier. To use this
classifier first we should clean dataset and apply preprocessing step then, after extracting features
of dataset I split **80%** train and **20%** test data in the given dataframe `train_test_split()` in `sklearn`
library. Before training the model, I used `MinMaxScaler()` in `sklearn.preprocessing` which is
used to transform features by scaling each feature to a given range. Normalization involves
adjusting values that exist on different scales into a common scale, allowing them to be more
readily compared.

### Evaluation:
  * **Accuracy**: 0.9847533632286996
  * **Recall**: 0.9172413793103448
  * **Precision**: 0.9637681159420289
  * **F1 score**: 0.939929328621908
  
## Part 2: Sentimental LIAR detection (Applying Naïve Bayes on the second dataset)
### Preprocess: 
  * **Tokenizing** by `word_tokenize()` in `nltk` library.
  * **Normalizing** and convert all letters to lower case by `lower()`, and removing punctuations.
  * **Removing stop words** like prepositions detecting by `nltk_stopwords()`.
  * **Lemmatizing** tokens by `WordNetLemmatizer()` in `nltk`.
  
### Feature Extraction:
Feature engineering is one of the most important steps in machine learning. It is the process of using domain knowledge of the data to create features that make machine learning algorithms work. If we can use suitable features and feed them to our model then the model will be able to understand the sentence better. Beside text we can use other sentiment columns as features. I employed correlation to detect relation between “label” and other features. Here is the plot of correlation matrix:

<p align="center">
    <img src="https://user-images.githubusercontent.com/69076293/227918203-f9723a10-b2d1-4130-9c61-52c8062564f5.png" alt="correlation matrix">
</p>

I extracted features based on [this paper](https://dl.acm.org/doi/pdf/10.5555/1667583.1667679) and [this link](https://research.signal-ai.com/assets/Deception_Detection_with_NLP.pdf):
  * **Frequency of words**: we use bag of words model. We use the assumption of independency and count the number of words in a record.
  * **Number of intensifier words**: intensifiers like “very”, “surely and etc can be a feature to detect liying.
  * **Number of words in OTHER class(like she, he, …)**: according to the paper, desceptive messages has more of these words.
  * **Number of words in Self class(like I, mine, …)**: according to the paper, truthful messages has more of these words.
  * **Positivity or negativity of sentiment_score**: we can see whether a sentiment is positive or negative.
  * **Range of 5 sentiments(“fear”, “joy”, “anger”, “disgust”, “sad”)**: the values of these 5 columns are a number between zero and one. Because of the fact we are looking for conditional probabilities in naïve bayes, the occurrence probability of a exact number is zero, so I considered a range for these columns. This range is [0, medianOfColumn) and [medianOfColumn, 1]. If value of a sentiment for a record is less than median the value is set to 0 and if not, the value is set to 1. Now we can extract some features of these columns which can be used by naïve bayes classifier.

### Training the model:
I employed `MultinomialNB()` in `sklearn` library to train naïve bayes classifier. To use this
classifier first we should clean dataset and apply preprocessing step then, after extracting features
of dataset I split **80%** train and **20%** test data in the given dataframe `train_test_split()` in `sklearn`
library. Before training the model, I used `MinMaxScaler()` in `sklearn.preprocessing` which is
used to transform features by scaling each feature to a given range. Normalization involves
adjusting values that exist on different scales into a common scale, allowing them to be more
readily compared.

### Evaluation:
  * **Accuracy**: 0.6179952644041041
  * **Recall**: 0.41229656419529837
  * **Precision**: 0.5891472868217055
  * **F1 score**: 0.4851063829787234
