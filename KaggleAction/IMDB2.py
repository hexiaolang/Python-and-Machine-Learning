# -*- coding:utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import feature_selection
import numpy as np
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
import xgboost as xgb

if __name__ == '__main__':
    train = pd.read_csv('../Datasets/IMDB/labeledTrainData.tsv',
                        delimiter='\t')
    test = pd.read_csv('../Datasets/IMDB/testData.tsv',
                       delimiter='\t')


    def review_to_text(review, remove_stopwords):
        # 去掉html标记
        raw_text = BeautifulSoup(review, 'html').get_text()
        # 去掉非字符字母
        letters = re.sub('[^a-zA-Z]', ' ', raw_text)
        words = letters.lower().strip()
        # 如果remove_stopwords被激活，则去停用词
        if remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = [w for w in words if w not in stop_words]
        return words


    # 预处理训练集和测试集数据
    # X_train = []
    # for review in train['review']:
    #     X_train.append(''.join(review_to_text(review, True)))
    X_test = []
    for review in test['review']:
        X_test.append(''.join(review_to_text(review, True)))
    y_train = train['sentiment']

    # 特征提取
    tfidf_vec = TfidfVectorizer(analyzer='word', stop_words='english')
    # X_train = tfidf_vec.fit_transform(X_train)
    X_test = tfidf_vec.fit_transform(X_test)

    # # 特征筛选
    # # 通过交叉验证，分析性能随特征筛选比例的变化
    # percentiles = range(1, 100, 2)
    # results = []
    #
    # for i in percentiles:
    #     fs = feature_selection.SelectPercentile(feature_selection.chi2,
    #                                             percentile=i)
    #     X_train_fs = fs.fit_transform(X_train, y_train)
    #     scores = cross_val_score(dt, X_train_fs, y_train, cv=5)
    #     results = np.append(results, scores.mean())
    # print results
    #

    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
    #                                                   test_size=.3, random_state=1)
    # xgb_val = xgb.DMatrix(X_val, label=y_val)
    # xgb_train = xgb.DMatrix(X_train, label=y_train)
    xgb_test = xgb.DMatrix(X_test)

    params = {
        'booster': 'gbtree',
        'objective': 'multi:softmax',  # 多分类的问题
        'num_class': 10,  # 类别数，与 multisoftmax 并用
        'gamma': 0.2,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 8,  # 构建树的深度，越大越容易过拟合
        'lambda': 3,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        'colsample_bytree': 0.5,  # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.01,  # 如同学习率
        'seed': 1000,
        'nthread': 8,  # cpu 线程数
        # 'eval_metric': 'auc'
    }

    plst = list(params.items())
    num_rounds = 500  # 迭代次数
    #
    # watchlist = [(xgb_train, 'train'), (xgb_val, 'val')]
    # model = xgb.train(plst, xgb_train, num_rounds, watchlist, early_stopping_rounds=100)
    #
    # model.save_model('../Datasets/IMDB/xgb.model')  # 用于存储训练出的模型

    # 模型预测
    model = xgb.Booster()
    model.load_model('../Datasets/IMDB/xgb.model')
    preds = model.predict(xgb_test)
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': preds})
    output.to_csv('../Datasets/IMDB/submission_xgb.csv', index=False,
                  quoting=3)
