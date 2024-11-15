### Gold: 
**You are given a dataset (named imdb_reviews.csv on Learning Central) with 
movie reviews and their associated sentiments. Your goal is to train machine 
learning models in the training set to predict the sentiment of a review in the test 
set. The problem should be framed as both a regression and a classification 
problem. The task is therefore to train two machine learning models (a regression 
and a classification model) and check their performance.**

## First step : Import the necessary libraries
- **Pandas**: Used to data analysis. It provides DataFrame and Series structures for easy data reading, cleaning and analysis.

- **Numpy**: A library for scientific computing, providing multi-dimensional array objects, matrix operations, linear algebra functions, and more.

- **Rs**: Python's regular expression library for pattern matching and replacement of strings.

- **Matplotlib**: A library for drawing various kinds of graphs, and pyplot is a submodule of it for simply drawing graphs.

import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error
from nltk.corpus import stopwords
import chardet
import nltk


#

import pandas as pd 

#

import  as np

#

import re

import string

#

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from collections import Counter

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from nltk.corpus import stopwords

import chardet

import nltk

