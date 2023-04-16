import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import warnings
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data

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
tfidf = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", stop_words=stopwords)
X = np.array(sms.msg_new)
Y = np.array(sms.label)
scales=MinMaxScaler(feature_range=(0,1))
X = tfidf.fit_transform(X).toarray()
X = scales.fit_transform(X)


# —————————————搭建全连接神经网络————————————
class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica,self).__init__()
        #定义第一个隐藏层
        self.hidden1=nn.Sequential(
            nn.Linear(
                in_features=57,#第一个隐藏层的输入，数据的特征数
                out_features=30,#第一个隐藏层的输出，神经元的数量
                bias=True,#默认设置偏置项
            ),
            nn.ReLU()
        )
        #定义第二个隐藏层
        self.hidden2=nn.Sequential(
            nn.Linear(30,10),#10个神经元
            nn.ReLU()
        )
        #分类层 二分类
        self.classifica=nn.Sequential(
            nn.Linear(10,2),#两个神经元
            nn.Sigmoid()
        )
    def forward(self,x):
        fc1=self.hidden1(x)
        fc2=self.hidden2(fc1)
        prob = nn.Softmax(fc2)
        output=self.classifica(fc2)
        return fc1,fc2,prob,output



# ———————————————模型训练————————————————
MyConvnet=MLPclassifica()
X_train_t=torch.from_numpy(X.astype(np.float32))
y_train_t=torch.from_numpy(Y.astype(np.int64))
train_data=Data.TensorDataset(X_train_t,y_train_t)
train_loader=Data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)
optimizer=torch.optim.Adam(MyConvnet.parameters(),lr=0.01)
loss_func=nn.CrossEntropyLoss() #交叉熵损失函数
for epoch in range(25):
    for step,(b_x,b_y) in enumerate(train_loader):
        _,_,_,output=MyConvnet(b_x)
        train_loss=loss_func(output,b_y)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

module_path = 'results/neural.model'
torch.save(MyConvnet.state_dict(),module_path)