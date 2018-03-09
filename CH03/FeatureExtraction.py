# -*- coding:utf-8 -*-
from sklearn.feature_extraction import DictVectorizer

measurements =[{'city': 'Dubai', 'temperature': 33.},
               {'city': 'London', 'temperature': 12.},
               {'city': 'San Fransisco', 'temperature': 18.}]
# 初始化特征抽取器
vec = DictVectorizer()
print vec.fit_transform(measurements).toarray()
print vec.get_feature_names()