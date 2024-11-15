
# Writing a System description Paper for [SemEval 2025 Task 11](https://github.com/emotion-analysis-project/SemEval2025-Task11)

_The content blogpost is based on content that you will find in the references as well as feedback received when running paper writing tutorials for SemEval2023 Task 12 and SemEval2024 Task 1._

_Some of these points are general so they can be applied to any paper you write._

In order to be included in our official ranking, you need to write a system description paper.  

**Note that you will not have to pay any fees for your paper to get published unless you would like to attend the SemEval workshop. In this case, you will have to check the website of the conference with which SemEval will be co-located.**


## First step : Import the necessary libraries

#Pandas used to data analysis. It provides DataFrame and Series structures for easy data reading, cleaning and analysis.

import pandas as pd 

#Library for scientific computing, providing multi-dimensional array objects, matrix operations, linear algebra functions, and more.

import numpy as np

#Python's regular expression library for pattern matching and replacement of strings.

import re

import string

#Matplotlib is a library for drawing various kinds of graphs, and pyplot is a submodule of it for simply drawing graphs.

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from collections import Counter

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, mean_squared_error

from nltk.corpus import stopwords

import chardet

import nltk

