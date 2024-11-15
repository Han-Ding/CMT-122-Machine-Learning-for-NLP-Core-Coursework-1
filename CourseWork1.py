# Import the necessary libraries
import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from nltk.corpus import stopwords
import chardet
import nltk
from sklearn.preprocessing import StandardScaler

# Download the list of deactivated words
# nltk.download('stopwords')

# Detecting file encoding
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)

# Read the file using the detected encoding
encoding = result['encoding']
df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', encoding=encoding)

# There are two columns in the dataset: ‘Reviews’ and ‘Sentiment’
X = df['Reviews']
y = df['Sentiment']

# Convert sentiment labels to binary labels: ‘pos’ -> 1, ‘neg’ -> 0
y = y.apply(lambda x: 1 if x == 'pos' else 0)

# Text Preprocessing Functions
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Strip out the numbers
    text = re.sub(r'\d+', '', text)
    # Participle
    words = text.split()
    # Disjunction
    stopwords_set = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwords_set]
    return words

# Pre-process all comments
X_processed = X.apply(preprocess_text)

# Statistical word frequency
all_words = [word for review in X_processed for word in review]
word_freq = Counter(all_words)

# Select the 5000 words that occur most frequently
most_common_words = [word for word, freq in word_freq.most_common(5000)]

# Convert text to feature vectors
def text_to_vector(text, vocabulary):
    vector = np.zeros(len(vocabulary))
    word_count = Counter(text)
    for idx, word in enumerate(vocabulary):
        vector[idx] = word_count[word]
    return vector

# Converting training and test sets into feature vectors
X_vectors = X_processed.apply(lambda x: text_to_vector(x, most_common_words))
X_vectors = np.stack(X_vectors)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# Training the regression model (linear regression)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction using regression modelling
y_pred_reg = regressor.predict(X_test)

# Calculate the root mean square error (RMSE) of the regression model
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print(f'Root mean square error of regression model(RMSE):{rmse}')

# Training classification models (logistic regression)
classifier = LogisticRegression(max_iter=1000)

# Normalising feature data to improve model convergence performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training using normalised data
classifier.fit(X_train_scaled, y_train)

# Prediction using classification models
y_pred_class = classifier.predict(X_test_scaled)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f'confusion matrix: \n{conf_matrix}')

# Calculation of accuracy, macro-average precision, recall and F1 score for classification models
accuracy = accuracy_score(y_test, y_pred_class)
macro_precision = precision_score(y_test, y_pred_class, average='macro')
macro_recall = recall_score(y_test, y_pred_class, average='macro')
macro_f1 = f1_score(y_test, y_pred_class, average='macro')

print(f'Accuracy of classification models: {accuracy}')
print(f'Macro-average accuracy of classification models: {macro_precision}')
print(f'Macro-mean recall for classification models: {macro_recall}')
print(f'Macro-averaged F1 scores for classification models: {macro_f1}')

# Visual Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictive labelling')
plt.ylabel('Actual label')
plt.title('Confusion matrix')
plt.show()

# Reset the training set indexes to ensure that the X_train and y_train indexes are the same
X_train_reset = pd.Series([X_processed[i] for i in y_train.index]).reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

# Statistical training focuses on positive and negative words
positive_reviews = X_train_reset[y_train_reset == 1]
negative_reviews = X_train_reset[y_train_reset == 0]

positive_word_freq = Counter([word for review in positive_reviews for word in review])
negative_word_freq = Counter([word for review in negative_reviews for word in review])

# Calculate the probability of each word
positive_word_prob = {word: freq / sum(positive_word_freq.values()) for word, freq in positive_word_freq.items()}
negative_word_prob = {word: freq / sum(negative_word_freq.values()) for word, freq in negative_word_freq.items()}

# Save results to a text file
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\positive_words.txt', 'w', encoding='utf-8') as f:
    for word, prob in positive_word_prob.items():
        f.write(f'{word}: {prob}\n')

with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\negative_words.txt', 'w', encoding='utf-8') as f:
    for word, prob in negative_word_prob.items():
        f.write(f'{word}: {prob}\n')

# Detailed step-by-step explanation:
# 1. Import necessary libraries: we use pandas to process the data and sklearn for data splitting, word frequency statistics and training the model.
# 2. Load dataset: pandas is used to read the CSV file, assuming that the dataset contains the columns ‘Reviews’ and ‘Sentiment’.
# 3. data preprocessing:
# - Convert reviews (Reviews) to lower case, remove punctuation and stop words, perform word splitting.
# - Remove English stop words using the stop word list in the nltk library.
# - Count the word frequency of all words and select the 5000 most frequent words for the glossary.
# - Convert each comment into a vector, the length of the vector is the size of the vocabulary list, and each element represents the number of times the word appears in the comment.
# 4. Train the regression model: train the training set using a linear regression model and make predictions on the test set, calculate the root mean square error (RMSE) to evaluate the performance of the regression model.
# 5. Train the classification model: use the logistic regression model to train on the training set and predict on the test set.
# 6. output the confusion matrix and compute the accuracy, macro-mean precision, recall and F1 score of the classification model to evaluate the performance of the model.
# 7. visualise the confusion matrix using seaborn and matplotlib to show the performance of the classification model more visually.
# 8. count the words in the positive and negative comments and calculate the probability of each word and store it in a local text file.

