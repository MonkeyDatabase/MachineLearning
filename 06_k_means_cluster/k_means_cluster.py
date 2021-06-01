import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import time

BLACK_LIST_TITLE = ['CUST_ID']


def load_data_frame(file_name)->pd.DataFrame:
    '''
    加载数据
    :param file_name: 文件路径
    :return: 非空dataframe
    '''
    df = pd.read_csv(file_name)
    # 去掉黑名单中的属性
    for black in BLACK_LIST_TITLE:
        df.drop(black, axis=1, inplace=True)
    # 用均值填充空值
    null_count = df.isnull().sum()
    for index in null_count.index:
        if null_count[index] != 0:
            df[index].fillna(df[index].mean(), inplace=True)
    return df


def corr(df:pd.DataFrame):
    corr = df.corr()
    top_features = corr.index
    plt.figure(figsize=(5, 5))
    sns.heatmap(df[top_features].corr(), annot=True)
    plt.show()


def preprocess(df:pd.DataFrame)->pd.DataFrame:
    df = pd.DataFrame(stats.zscore(df), columns=df.columns)
    return df


def visual_after_preprocess(df:pd.DataFrame):
    plotnumber = 1
    for i in df:
        if plotnumber > 15:
            break
        ax = plt.subplot(10, 2, plotnumber)
        sns.kdeplot(i)
        plotnumber+=1
    plt.show()


def rand_centers_of_mass(df:pd.DataFrame, k=5)->pd.DataFrame:
    df_centers = pd.DataFrame(index=range(k), columns=df.columns)
    for column in df.columns:
        min_value = df[column].min()
        max_value = df[column].max()
        range_column = max_value - min_value
        df_centers[column] = min_value + np.random.rand(k, 1) * range_column
    print('---------------------------------init center---------------------------------')
    print(df_centers)
    print('-----------------------------------------------------------------------------')
    return df_centers


def calc_dist(XA:pd.Series, XB:pd.Series)->float:
    if XA.shape != XB.shape:
        return -1
    size, = XA.shape
    dist_2 = 0
    for i in range(size):
        # print(XA[i], XB[i], XA[i]-XB[i],dist_2)
        dist_2 += (XA[i] - XB[i]) ** 2
    return np.sqrt(dist_2)


def k_means(df:pd.DataFrame, k=5):
    count = 0
    cluster_assemble = pd.DataFrame(index=df.index, columns=['iloc', 'distance'])
    cluster_assemble.loc[:, :] = 0
    centers_of_mass = rand_centers_of_mass(df, k)
    cluster_changed_flag = True
    while cluster_changed_flag:
        count += 1
        print('-------------------------------------{}starts-------------------------------------'.format(count))
        cluster_changed_flag = False
        for node in df.index:
            min_dist = np.inf
            min_index = -1
            for center in centers_of_mass.index:
                a = df.iloc[node]
                b = centers_of_mass.iloc[center]
                distance = calc_dist(a, b)
                if distance < min_dist:
                    min_dist = distance
                    min_index = center
            if cluster_assemble.iloc[node][0] != min_index:
                cluster_changed_flag = True
            cluster_assemble.iloc[node] = [min_index, min_dist]

        for center in centers_of_mass.index:
            #print(centers_of_mass)
            nodes_in_my_cluster = cluster_assemble[(cluster_assemble['iloc'] == center)].index
            print('cluster{}：{}'.format(center, nodes_in_my_cluster))
            if df.iloc[nodes_in_my_cluster].shape[0] != 0:
                centers_of_mass.iloc[center] = df.iloc[nodes_in_my_cluster].mean()
            else:
                # 当随机生成的质心周围无数据点时，重新为其随机
                centers_of_mass.iloc[center] = rand_centers_of_mass(df, 1).iloc[0]
            #print(centers_of_mass)

        print('-------------------------------------{}end-------------------------------------'.format(count))
    return cluster_assemble, centers_of_mass




if __name__ == '__main__':
    df = load_data_frame(r'./dataset/CC_GENERAL.csv')
    df = preprocess(df)
    cluster_assemble, centers_of_mass = k_means(df, 5)
    print('-------------------------------------final assemble-------------------------------------')
    print(cluster_assemble)
    cluster_assemble.to_csv("./output/cluster_assemble_{}.csv".format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))))
    print('-------------------------------------final center-------------------------------------')
    print(centers_of_mass)
    centers_of_mass.to_csv("./output/centers_of_mass_{}.csv".format(time.strftime('%Y-%m-%d %H-%M-%S', time.localtime(time.time()))))