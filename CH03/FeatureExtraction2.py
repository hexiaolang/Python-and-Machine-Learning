# -*- coding:utf-8 -*-
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

news = fetch_20newsgroups(subset='all')

X_train, X_test, y_train, y_test = train_test_split(news.data,
                                                    news.target,
                                                    test_size=.25,
                                                    random_state=33)

# 特征抽取
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word',
                                                     stop_words='english'),\
    TfidfVectorizer(analyzer='word', stop_words='english')
X_count_train = count_filter_vec.fit_transform(X_train)
X_count_test = count_filter_vec.transform(X_test)
X_tfidf_train = tfidf_filter_vec.fit_transform(X_train)
X_tfidf_test = tfidf_filter_vec.transform(X_test)

# 建模评估
mnb_count_filter = MultinomialNB()
mnb_count_filter.fit(X_count_train, y_train)
y_count_filter_pred = mnb_count_filter.predict(X_count_test)

mnb_tfidf_filter = MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_train, y_train)
y_tfidf_filter_pred = mnb_tfidf_filter.predict(X_tfidf_test)

print 'The accuracy of classifying 20newsgroups using Naive Bayes' \
      '(CountVectorizer by filtering stopwords):', \
    mnb_count_filter.score(X_count_test, y_test)

print 'The accuracy of classifying 20newsgroups using Naive Bayes' \
      '(TfidfVectorizor by filtering stopwords):',\
    mnb_tfidf_filter.score(X_tfidf_test, y_test)

print classification_report(y_test, y_count_filter_pred,
                            target_names=news.target_names)
print classification_report(y_test, y_tfidf_filter_pred,
                            target_names=news.target_names)



