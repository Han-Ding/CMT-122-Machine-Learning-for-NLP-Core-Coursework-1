## _**Goal**._
**You are given a dataset (named imdb_reviews.csv on Learning Central) with 
movie reviews and their associated sentiments. Your goal is to train machine 
learning models in the training set to predict the sentiment of a review in the test 
set. The problem should be framed as both a regression and a classification 
problem. The task is therefore to train two machine learning models (a regression 
and a classification model) and check their performance.**

---

## _**Main ideas**._
1. Import necessary libraries: we use pandas to process the data and sklearn for data splitting, word frequency statistics and training the model.
2. Load dataset: pandas is used to read the CSV file, assuming that the dataset contains the columns ‘Reviews’ and ‘Sentiment’.
3. data preprocessing:
   - Convert reviews (Reviews) to lower case, remove punctuation and stop words, perform word splitting.
   - Remove English stop words using the stop word list in the nltk library.
   - Count the word frequency of all words and select the 5000 most frequent words for the glossary.
   - Convert each comment into a vector, the length of the vector is the size of the vocabulary list, and each element represents the number of times the word appears in the comment.
4. Train the regression model: train the training set using a linear regression model and make predictions on the test set, calculate the root mean square error (RMSE) to evaluate the performance of the regression model.
5. Train the classification model: use the logistic regression model to train on the training set and predict on the test set.
6. Output the confusion matrix and compute the accuracy, macro-mean precision, recall and F1 score of the classification model to evaluate the performance of the model.
7. Visualise the confusion matrix using seaborn and matplotlib to show the performance of the classification model more visually.
8. Count the words in the positive and negative comments and calculate the probability of each word and store it in a local text file.

---
 
## _First step: Import the necessary libraries._
- **Pandas**: Used to data analysis. It provides DataFrame and Series structures for easy data reading, cleaning and analysis.

- **Numpy**: A library for scientific computing, providing multi-dimensional array objects, matrix operations, linear algebra functions, and more.

- **Rs**: Python's regular expression library for pattern matching and replacement of strings.

- **Matplotlib**: A library for drawing various kinds of graphs, and pyplot is a submodule of it for simply drawing graphs.

- **Seaborn**: Advanced plotting library based on matplotlib for more intuitive and aesthetically pleasing plots, especially for statistical charts.

- **Train_test_split**: Used to split the dataset into a training set and a test set.

- **Counter**: Used to count the frequency of elements in an iterable object.

- **LinearRegression, LogisticRegression**: A linear model in the scikit-learn library for regression and classification tasks.

- **accuracy_score、precision_score、recall_score、f1_score**: Used to evaluate the performance of the classification model (accuracy, precision, recall, F1 score).

- **stopwords**: Used to remove stop words from text

- **chardet**: Used to detect the encoding format of CSV files so that files containing special characters can be read correctly.

- **nltk**: A common library for natural language processing, containing a rich set of tools and datasets.

---

## _**Second step: Load dataset**._
- Detecting file encoding
```
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', 'rb') as file:
      result = chardet.detect(file.read(100000))
      print(result)
```
_**Note: The 100000 here is so that all the data can be read in.**._

- Read the file using the detected encoding
```
encoding = result['encoding']
df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', encoding=encoding)
```
_**Note: Here there may be an error ‘UnicodeEncodeError, python’, because the default encoding format when python writes text to an external file is GBLK, 
normally you can specify the encoding format, here's how to let python recognise the file encoding itself.**._

- Separate feature and target columns in the dataset for the following vectorisation

```
X = df['Reviews']
y = df['Sentiment']
```

- Convert sentiment tags to binary tags, because most machine learning models (especially logistic regression, linear regression, etc.) can only handle numeric types of data, not strings directly.

```
y = y.apply(lambda x: 1 if x == 'pos' else 0)
```
---

## _**Third Step: Text Preprocessing Functions**._
```
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
```
**_Note: ._**
- _**It is important to note that the removal of punctuation is a function of string, and that maketrans is designed to create a character mapping table for easy translating**._
- _**Removing deactivated words is nltk's own deactivation table.**._

```
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
```
_**Note: The point of counting word frequencies here is to select meaningful words, ignoring irrelevant and few words,
        and to reduce the dimensionality of the calculation, but of course one can always adjust the range of this word frequency, 
        and a wider range may also lead to a higher degree of accuracy**._
        
---

## _**Forth Step: The original data be separated into two distinct sets: one for training and one for testing**._
```
# Converting training and test sets into feature vectors
X_vectors = X_processed.apply(lambda x: text_to_vector(x, most_common_words))
X_vectors = np.stack(X_vectors)

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)
```
**_Note: The values of test_size and random_state can be changed here._**
- _**test_size: Controlling the size of train data and test data**._
- _**random_state: Controlling the seed of a random number generator. (42 is a common convention)**._

---

## _**Fifth Step: Regression model**._

```
# Training regression models (linear regression)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction using regression models
y_pred_reg = regressor.predict(X_test)

# Calculate the root mean square error (RMSE) of the regression model
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print(f'Root mean square error of regression model（RMSE）：{rmse}')
```
**_Note: RMSE is a commonly used metric in regression tasks, measuring the average degree of deviation between predicted and actual values._**
- **_The smaller the value, the smaller the prediction error of the model and the better the performance of the model._**
- **_If the RMSE value is large, it means that the model's prediction ability is not good and the error is large._**

---

## _**Sixth Step: Classification model**._

```
# Training classification models (logistic regression)
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# Prediction using classification models
y_pred_class = classifier.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f'confusion matrix: \n{conf_matrix}')
```
**_Note: The confusion matrix is commonly used to test the accuracy of categorical regression models, and it is more intuitive to see the model's predictions when it is visualised graphically._**

```
# Calculation of accuracy, macro-average precision, recall and F1 score for classification models
accuracy = accuracy_score(y_test, y_pred_class)
macro_precision = precision_score(y_test, y_pred_class, average='macro')
macro_recall = recall_score(y_test, y_pred_class, average='macro')
macro_f1 = f1_score(y_test, y_pred_class, average='macro')

print(f'Accuracy of classification models：{accuracy}')
print(f'Macro average precision of classification model: {macro_precision}')
print(f'Macro average recall of classification models: {macro_recall}')
print(f'Macro average F1 score of classification models: {macro_f1}')

# Visual Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictive labelling')
plt.ylabel('Actual label')
plt.title('Confusion matrix')
plt.show()
```
**_Note: ._**
- **_figsize controls the length and width of Matrix ._**
- **_annot=True shows the numbers._**
- **cmap='Blues' means color is bule._**
- **fmt='g' means displaying values formatted as integers._**
- **Matplotlib's default fonts do not support some Chinese characters._**
- **_Extral attention: Here plt.show() displays the confusion matrix I have placed it below the calculation of the individual values because the code performing this operation must manually switch off the confusion matrix display legend before it can proceed.
      So far I haven't found a solution to skip this step and perform it all._**

---

## _**Seventh Step: Counting words in positive and negative comments and storing it in a local text file**._

```
# Reset the training set indexes to ensure that the X_train and y_train indexes are the same
X_train_reset = pd.Series([X_processed[i] for i in y_train.index]).reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)
```
**_Note: Here may has a IndexingError. This is one of the most difficult and confusing points of all the code errors for me to understand.
The reason is the indexes of X_train and y_train are inherited from the original data, so their indexes will not match the original X_processed._**

- _**Pandas Series: because both X_train and X_processed are initially Pandas objects, and using a Series makes it easier to follow up.**._
- _**reset_index(drop=True)： Reset the index to generate consecutive indexes starting from 0.
drop=True: indicates that the old index is not kept after resetting the index (the old index is no longer useful to us).**._

```
# Statistical training focuses on positive and negative words
positive_reviews = X_train_reset[y_train_reset == 1]
negative_reviews = X_train_reset[y_train_reset == 0]

positive_word_freq = Counter([word for review in positive_reviews for word in review])
negative_word_freq = Counter([word for review in negative_reviews for word in review])

# Calculate the probability of each word
positive_word_prob = {word: freq / sum(positive_word_freq.values()) for word, freq in positive_word_freq.items()}
negative_word_prob = {word: freq / sum(negative_word_freq.values()) for word, freq in negative_word_freq.items()}

# Save results to a text file
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\positive_words.txt', 'w') as f:
    for word, prob in positive_word_prob.items():
        f.write(f'{word}: {prob}\n')

with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\negative_words.txt', 'w') as f:
    for word, prob in negative_word_prob.items():
        f.write(f'{word}: {prob}\n')
```

---

## _**Conclusion**._
**_Generally, this is not a complex code to understand. The most of funcation can be used directly through embedded function in python. 
But I also got a first-hand sense of how difficult it is to be a postgraduate student, as this small assignment was much the same as my undergraduate thesis, which was a sentiment analysis of data.
The code definitely has shortcomings and needs to be improved. I need to learn more._**























