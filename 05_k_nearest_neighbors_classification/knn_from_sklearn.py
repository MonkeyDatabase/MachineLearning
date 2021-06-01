import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 读取数据集
train_set = pd.read_csv('./dataset/fruit_data_with_colors.txt', sep='	')
train = pd.concat([train_set['mass'], train_set['width'], train_set['height'], train_set['color_score']], axis=1)
train = np.array(train)
scaler = StandardScaler()
train = scaler.fit_transform(train)
label = np.array(train_set['fruit_label'])

# 切分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(train, label, train_size=0.8, random_state=1)

# 调包，建立KD树
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)

# 预测
y_perdict = knn.predict(x_test)
print(accuracy_score(y_test, y_perdict))