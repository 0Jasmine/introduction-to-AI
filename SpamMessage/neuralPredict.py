import torch
from module import MLPclassifica,tfidf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
scales=MinMaxScaler(feature_range=(0,1))
module_path = 'results/neural.model'
MyConvnet = MLPclassifica()
MyConvnet.load_state_dict(torch.load(module_path))
MyConvnet.eval()


def predict(message:str):
    """
    预测短信短信的类别和每个类别的概率
    param: message: 经过jieba分词的短信，如"医生 拿 着 我 的 报告单 说 ： 幸亏 你 来 的 早 啊"
    return: label: 整数类型，短信的类别，0 代表正常，1 代表恶意
            proba: 列表类型，短信属于每个类别的概率，如[0.3, 0.7]，认为短信属于 0 的概率为 0.3，属于 1 的概率为 0.7
    """  
    words = tfidf.get_feature_names()
    X =  [0 for _ in range(len(words))]
    message = message.split(' ')
    for word in message:
        for index,wword in enumerate(words):
            if word==wword:
                X[index]+=1
    X = scales.fit_transform(X)
    _,_,proba,label = MyConvnet(X)
    return label, proba