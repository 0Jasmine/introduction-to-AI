from main import *
import matplotlib as plt
file_dir = './data'
csv_files = os.listdir(file_dir)
# df 作为最后输出的 DataFrame 初始化为空
df = pd.DataFrame()
feature = ['cpc', 'cpm']
df_features = []
for col in feature:
    infix = col + '.csv'
    path = os.path.join(file_dir, infix)
    df_feature = pd.read_csv(path)
    # 将两个特征存储起来用于后续连接
    df_features.append(df_feature)

# 2 张 DataFrame 表按时间连接
df = pd.merge(left=df_features[0], right=df_features[1])

data = preprocess_data(df)
is_anomaly, preprocess_data, kmeans, ratio = predict(data)
a = df.loc[is_anomaly['is_anomaly'] == 1, ['timestamp', 'cpc']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpc'], color='blue')
# 聚类后 cpc 的异常点
plt.scatter(a['timestamp'],a['cpc'], color='red')
plt.show()

a = df.loc[is_anomaly['is_anomaly'] == 1, ['timestamp', 'cpm']] 
plt.figure(figsize=(20,6))
plt.plot(df['timestamp'], df['cpm'], color='blue')
# 聚类后 cpm 的异常点
plt.scatter(a['timestamp'],a['cpm'], color='red')
plt.show()