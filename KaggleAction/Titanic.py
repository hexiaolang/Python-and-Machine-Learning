# -*— coding:utf-8 -*-
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import GridSearchCV

if __name__ == '__main__':
    train = pd.read_csv('../Datasets/Titanic/train.csv')
    test = pd.read_csv('../Datasets/Titanic/test.csv')
    # 根据经验，人工选取有效特征
    selected_features = ['Pclass', 'Sex', 'Age', 'Embarked',
                         'SibSp', 'Parch', 'Fare']
    X_train = train[selected_features]
    X_test = test[selected_features]
    y_train = train['Survived']
    # 填充缺失值
    X_train['Embarked'].fillna('S', inplace=True)
    X_test['Embarked'].fillna('S', inplace=True)
    X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
    X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
    X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

    dict_vec = DictVectorizer(sparse=False)
    X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = dict_vec.transform(X_test.to_dict(orient='record'))


    # 建模
    rfc = RandomForestClassifier()
    xgbc = XGBClassifier()
    # 使用5折交叉验证进行性能评估
    print cross_val_score(rfc, X_train, y_train, cv=5).mean()
    print cross_val_score(xgbc, X_train, y_train, cv=5).mean()

    rfc.fit(X_train, y_train)
    rfc_y_pred = rfc.predict(X_test)
    rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                   'Survived': rfc_y_pred})
    rfc_submission.to_csv('../Datasets/Titanic/rfc_submission.csv',
                          index=False)

    xgbc.fit(X_train, y_train)
    xgbc_y_pred = xgbc.predict(X_test)
    xgbc_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                   'Survived': xgbc_y_pred})
    xgbc_submission.to_csv('../Datasets/Titanic/xgbc_submission.csv',
                           index=False)

    # 并行网格搜索寻找更好的超参数组合，提高XGBoost预测性能
    params = {'max_depth':range(2, 7),
              'n_estimators':range(100, 1100, 200),
              'learning_rate':[0.05, 0.1, 0.25, 0.5, 1.0]}
    xgbc_best = XGBClassifier()
    gs = GridSearchCV(xgbc_best, params, n_jobs=-1, cv=5, verbose=0)
    gs.fit(X_train, y_train)
    xgbc_best_y_pred = gs.predict(X_test)
    xgbc_best_submission = pd.DataFrame({'PassengerId': test['PassengerId'],
                                         'Survived': xgbc_best_y_pred})
    xgbc_best_submission.to_csv('../Datasets/Titanic/xgbc_best_submission.csv',
                                index=False)


