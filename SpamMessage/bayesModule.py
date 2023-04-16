import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
import numpy as np

# ————————————————设置环境————————————————
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
warnings.filterwarnings('ignore')

# ———————————————读取停用词库——————————————
def read_stopwords(stopwords_path):
    """
    读取停用词库
    :param stopwords_path: 停用词库的路径
    :return: 停用词列表，如 ['嘿', '很', '乎', '会', '或']
    """
    stopwords = []
    stopwordsFile = open(stopwords_path)
    stopwords = stopwordsFile.readlines()
    stopwordsFile.close()
    return stopwords

stopwords_path = r'new_stopwords.txt'
stopwords = read_stopwords(stopwords_path)

# ———————————————读取、处理数据——————————————
data_path = "./datasets/5f9ae242cae5285cd734b91e-momodel/sms_pub.csv"
sms = pd.read_csv(data_path, encoding='utf-8')
X = np.array(sms.msg_new)
Y = np.array(sms.label)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=42, test_size=0.2)
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
scales=MinMaxScaler(feature_range=(0,1))


# ———————————————搭建、训练模型——————————————
pipeline_list = [
    ('tv', tfidf),
    ('sc',scales),
    ('classifier', MultinomialNB())
]
pipeline = Pipeline(pipeline_list)
pipeline.fit(X_train, Y_train)

# ———————————————保存模型——————————————
pipeline_path = 'results/pipeline.model'
joblib.dump(pipeline, pipeline_path)

