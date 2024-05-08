import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import naive_bayes
from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_auc_score


# 2a) feature preprocessing:
# i) takes in train and test sets and do nothing
def do_nothing(train, test):
    return train, test


# ii) fit a standard scaler on train set and apply on both train and test sets
def do_std(train, test):
    scaler = preprocessing.StandardScaler()
    new_train = scaler.fit_transform(train)
    new_test = scaler.transform(test)

    return new_train, new_test


# iii) transform train and test sets using log(x_ij + 0.1)
def do_log(train, test):
    scaler = preprocessing.FunctionTransformer(lambda x: np.log(x + 0.1))
    new_train = scaler.transform(train)
    new_test = scaler.transform(test)

    return new_train, new_test


# iv) binarize the features with a threshold of 0
def do_bin(train, test):
    scaler = preprocessing.Binarizer()
    new_train = scaler.transform(train)
    new_test = scaler.transform(test)

    return new_train, new_test


# 2b) Naive Bayes
def eval_nb(trainx, trainy, testx, testy):
    model = naive_bayes.GaussianNB()
    model.fit(trainx, trainy)

    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    train_prob = model.predict_proba(trainx)[:, 1]
    test_prob = model.predict_proba(testx)[:, 1]

    train_acc = accuracy_score(trainy, train_pred)
    test_acc = accuracy_score(testy, test_pred)
    train_auc = roc_auc_score(trainy, train_prob)
    test_auc = roc_auc_score(testy, test_prob)

    result = {
        'train-acc': train_acc,
        'train-auc': train_auc,
        'test-acc': test_acc,
        'test-auc': test_auc,
        'test-prob': test_prob
    }

    return result


# 2d) logistic regression
def eval_lr(trainx, trainy, testx, testy):
    model = linear_model.LogisticRegression(penalty=None, max_iter=4000)
    model.fit(trainx, trainy)

    train_pred = model.predict(trainx)
    test_pred = model.predict(testx)
    train_prob = model.predict_proba(trainx)[:, 1]
    test_prob = model.predict_proba(testx)[:, 1]

    train_acc = accuracy_score(trainy, train_pred)
    test_acc = accuracy_score(testy, test_pred)
    train_auc = roc_auc_score(trainy, train_prob)
    test_auc = roc_auc_score(testy, test_prob)

    result = {
        'train-acc': train_acc,
        'train-auc': train_auc,
        'test-acc': test_acc,
        'test-auc': test_auc,
        'test-prob': test_prob
    }

    return result
