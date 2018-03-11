# -*- coding:utf-8 -*-
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# Pipeline用于方便搭建系统流程
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
import nltk.data
from gensim.models import word2vec
from gensim.models import Word2Vec
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.grid_search import GridSearchCV

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
    X_train = []
    for review in train['review']:
        X_train.append(''.join(review_to_text(review, True)))
    X_test = []
    for review in test['review']:
        X_test.append(''.join(review_to_text(review, True)))
    y_train = train['sentiment']

    pip_count = Pipeline([('count_vec', CountVectorizer(analyzer='word')),
                          ('mnb', MultinomialNB())])
    pip_tfidf = Pipeline([('tfidf_vec', TfidfVectorizer(analyzer='word')),
                          ('mnb', MultinomialNB())])
    # 配置超参数搜索组合
    params_count = {'count_vec__binary': [True, False],
                    'count_vec__ngram_range': [(1, 1), (1, 2)],
                    'mnb__alpha':[0.1, 1.0, 10.0]}
    params_tfidf = {'tfidf_vec__binary': [True, False],
                    'tfidf_vec__ngram_range': [(1, 1), (1, 2)],
                    'mnb__alpha': [0.1, 1.0, 10.0]}
    # 采用4折交叉验证对使用CountVectorizer的朴素贝叶斯模型进行超参数搜索
    gs_count = GridSearchCV(pip_count, params_count, cv=4, n_jobs=-1)
    gs_count.fit(X_train, y_train)
    # 以最佳的超参数组合配置模型并预测
    count_y_pred = gs_count.predict(X_test)

    # 采用4折交叉验证对使用TfidfVectorizer的朴素贝叶斯模型进行超参数搜索
    gs_tfidf = GridSearchCV(pip_tfidf, params_tfidf, cv=4, n_jobs=-1)
    gs_tfidf.fit(X_train, y_train)
    tfidf_pred = gs_tfidf.predict(X_test)

    submission_count = pd.DataFrame({'id': test['id'], 'sentiment': count_y_pred})
    submission_tfidf = pd.DataFrame({'id': test['id'], 'sentiment': tfidf_pred})
    submission_count.to_csv('../Datasets/IMDB/submission_count.csv', index=False)
    submission_tfidf.to_csv('../Datasets/IMDB/submission_tfidf.csv', index=False)

    # 读取未标记数据
    unlabeled_train = pd.read_csv('../Datasets/IMDB/unlabeledTrainData.tsv',
                                  delimiter='\t', quoting=3)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # 逐条对影评分句
    def review_to_sentences(review, tokenizer):
        raw_sentences = tokenizer.tokenize(review.strip())
        sentences = []
        for raw_sentence in raw_sentences:
            if len(raw_sentence) > 0:
                sentences.append(review_to_text(raw_sentence, False))
        return sentences


    corpora = []
    # 准备用于训练词向量的数据
    for review in unlabeled_train['review']:
        corpora += review_to_sentences(review.decode('utf-8'), tokenizer)
    # 配置训练词向量模型的超参数
    num_features = 300
    min_word_count = 20
    num_workers = 4
    context = 10
    downsampling = 1e-3
    # 训练词向量模型
    model = word2vec.Word2Vec(corpora, workers=num_workers,
                              size=num_features,
                              min_count=min_word_count,
                              window=context,
                              sample=downsampling)
    model.init_sims(replace=True)
    model_name = '../Datasets/IMDB/300features_20minwords_10context'
    # 存储模型
    model.save(model_name)

    # 直接读入已经训练好的词向量模型
    model = Word2Vec.load('../Datasets/IMDB/300features_20minwords_10context')

    # 定义一个函数使用词向量产生文本特征向量
    def makeFeatureVec(words, model, num_features):
        featureVec = np.zeros((num_features,), dtype="float32")
        nwords = 0
        index2word_set = set(model.wv.index2word)
        for word in words:
            if word in index2word_set:
                nwords = nwords + 1.
                featureVec = np.add(featureVec, model[word])
        featureVec = np.divide(featureVec, nwords)
        return featureVec


    # 将每条影评转化为基于词向量的特征向量（平均词向量）
    def getAvgFeatureVecs(reviews, model, num_features):
        counter = 0
        reviewFeatureVecs = np.zeros((len(reviews), num_features),
                                     dtype='float32')
        for review in reviews:
            reviewFeatureVecs[counter] = makeFeatureVec(review,
                                                        model,
                                                        num_features)
        return reviewFeatureVecs


    # 准备新的基于词向量表示的训练和测试特征向量
    clean_train_reviews = []
    for review in train['review']:
        clean_train_reviews.append(review_to_text(review,
                                                  remove_stopwords=True))
    trainDataVecs = getAvgFeatureVecs(clean_train_reviews, model,
                                      num_features)

    clean_test_reviews = []
    for review in test['review']:
        clean_test_reviews.append(review_to_text(review,
                                                 remove_stopwords=True))
    testDataVecs = getAvgFeatureVecs(clean_test_reviews,
                                     model,
                                     num_features)

    gbc = GradientBoostingClassifier()
    params_gbc = {'n_estimators': [10, 100, 500],
                  'learning_rate': [0.01, 0.1, 1.0],
                  'max_depth': [2, 3, 4]}
    gs = GridSearchCV(gbc, params_gbc, cv=4, n_jobs=-1)
    gs.fit(trainDataVecs, y_train)
    result = gs.predict(testDataVecs)
    output = pd.DataFrame(data={'id': test['id'], 'sentiment': result})
    output.to_csv('../Datasets/IMDB/submission_w2v.csv', index=False,
                  quoting=3)