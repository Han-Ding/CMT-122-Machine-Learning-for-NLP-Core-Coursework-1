# 导入必要的库
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

# 下载停用词表
nltk.download('stopwords')

# 检测文件编码
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', 'rb') as file:
    result = chardet.detect(file.read(100000))
    print(result)

# 使用检测到的编码读取文件
encoding = result['encoding']
df = pd.read_csv(r'B:\CODE\PythonProject\122\COURSE WORK 1\imdb_reviews.csv', encoding=encoding)

# 数据集中有两列：'Reviews' 和 'Sentiment'
X = df['Reviews']
y = df['Sentiment']

# 将情感标签转换为二进制标签：'pos' -> 1，'neg' -> 0
y = y.apply(lambda x: 1 if x == 'pos' else 0)

# 文本预处理函数
def preprocess_text(text):
    # 转换为小写
    text = text.lower()
    # 去除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 去除数字
    text = re.sub(r'\d+', '', text)
    # 分词
    words = text.split()
    # 去除停用词
    stopwords_set = set(stopwords.words('english'))
    words = [word for word in words if word not in stopwords_set]
    return words

# 对所有评论进行预处理
X_processed = X.apply(preprocess_text)

# 统计词频
all_words = [word for review in X_processed for word in review]
word_freq = Counter(all_words)

# 选择出现频率最高的5000个词
most_common_words = [word for word, freq in word_freq.most_common(5000)]

# 将文本转换为特征向量
def text_to_vector(text, vocabulary):
    vector = np.zeros(len(vocabulary))
    word_count = Counter(text)
    for idx, word in enumerate(vocabulary):
        vector[idx] = word_count[word]
    return vector

# 将训练集和测试集转换为特征向量
X_vectors = X_processed.apply(lambda x: text_to_vector(x, most_common_words))
X_vectors = np.stack(X_vectors)

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.2, random_state=42)

# 步骤 3：训练回归模型（线性回归）
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# 使用回归模型进行预测
y_pred_reg = regressor.predict(X_test)

# 计算回归模型的均方根误差（RMSE）
rmse = np.sqrt(mean_squared_error(y_test, y_pred_reg))
print(f'回归模型的均方根误差（RMSE）：{rmse}')

# 步骤 4：训练分类模型（逻辑回归）
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

# 使用分类模型进行预测
y_pred_class = classifier.predict(X_test)

# 计算混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred_class)
print(f'混淆矩阵：\n{conf_matrix}')



# 计算分类模型的准确率、宏平均精度、召回率和F1分数
accuracy = accuracy_score(y_test, y_pred_class)
macro_precision = precision_score(y_test, y_pred_class, average='macro')
macro_recall = recall_score(y_test, y_pred_class, average='macro')
macro_f1 = f1_score(y_test, y_pred_class, average='macro')

print(f'分类模型的准确率：{accuracy}')
print(f'分类模型的宏平均精度：{macro_precision}')
print(f'分类模型的宏平均召回率：{macro_recall}')
print(f'分类模型的宏平均F1分数：{macro_f1}')

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predictive labelling')
plt.ylabel('Actual label')
plt.title('Confusion matrix')
plt.show()

# 重置训练集索引，确保 X_train 和 y_train 的索引一致
X_train_reset = pd.Series([X_processed[i] for i in y_train.index]).reset_index(drop=True)
y_train_reset = y_train.reset_index(drop=True)

# 统计训练集中积极和消极词汇
positive_reviews = X_train_reset[y_train_reset == 1]
negative_reviews = X_train_reset[y_train_reset == 0]

positive_word_freq = Counter([word for review in positive_reviews for word in review])
negative_word_freq = Counter([word for review in negative_reviews for word in review])

# 计算每个词的概率
positive_word_prob = {word: freq / sum(positive_word_freq.values()) for word, freq in positive_word_freq.items()}
negative_word_prob = {word: freq / sum(negative_word_freq.values()) for word, freq in negative_word_freq.items()}

# 将结果保存到文本文件
with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\positive_words.txt', 'w', encoding='utf-8') as f:
    for word, prob in positive_word_prob.items():
        f.write(f'{word}: {prob}\n')

with open(r'B:\CODE\PythonProject\122\COURSE WORK 1\negative_words.txt', 'w', encoding='utf-8') as f:
    for word, prob in negative_word_prob.items():
        f.write(f'{word}: {prob}\n')

# 详细步骤解释：
# 1. 导入必要的库：我们使用pandas处理数据，使用sklearn进行数据拆分、词频统计和训练模型。
# 2. 加载数据集：使用pandas读取CSV文件，假设数据集中包含'Reviews'和'Sentiment'两列。
# 3. 数据预处理：
#    - 将评论（Reviews）转换为小写，去除标点符号和停用词，进行分词。
#    - 使用nltk库中的停用词表去除英文停用词。
#    - 统计所有词的词频，选择出现频率最高的5000个词作为词汇表。
#    - 将每个评论转换为向量，向量长度为词汇表大小，每个元素表示该词在评论中的出现次数。
# 4. 训练回归模型：使用线性回归模型对训练集进行训练，并对测试集进行预测，计算均方根误差（RMSE）来评估回归模型的表现。
# 5. 训练分类模型：使用逻辑回归模型对训练集进行训练，并对测试集进行预测。
# 6. 输出混淆矩阵，并计算分类模型的准确率、宏平均精度、召回率和F1分数，以评估模型的性能。
# 7. 使用seaborn和matplotlib可视化混淆矩阵，以更直观地展示分类模型的性能。
# 8. 统计积极和消极评论中的词汇，并计算每个词汇的概率，将其存储到本地文本文件中。
