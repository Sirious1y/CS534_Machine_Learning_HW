import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score


# 3b) holdout
def generate_train_val(x, y, valsize):
    n = x.shape[0]
    val_n = int(n * valsize)

    indices = np.random.permutation(n)

    train_idx = indices[val_n:]
    val_idx = indices[:val_n]

    train_x = x[train_idx]
    train_y = y[train_idx]
    val_x = x[val_idx]
    val_y = y[val_idx]

    result = {
        'train-x': train_x,
        'train-y': train_y,
        'val-x': val_x,
        'val-y': val_y
    }

    return result


# 3c) k-fold
def generate_kfold(x, y, k):
    n = len(y)
    fold_size = n // k
    remainder = n - k * fold_size
    fold_idx = np.zeros(n, dtype=int)

    # indices = np.random.permutation(n)
    #
    # for i in range(k):
    #     for j in indices[i * fold_size : (i + 1) * fold_size]:
    #         fold_idx[j] = i
    #
    # for i in range(remainder):
    #     idx = indices[k * fold_size + i]
    #     fold_idx[idx] = i

    for i in range(k):
        fold_idx[i * fold_size : (i + 1) * fold_size] = i

    for i in range(remainder):
        fold_idx[fold_size * k + i] = i

    np.random.shuffle(fold_idx)

    return fold_idx


# 3d) evaluate holdout using logistic regression
def eval_holdout(x, y, valsize, logistic):
    split_data = generate_train_val(x, y, valsize)
    train_x = split_data['train-x']
    train_y = split_data['train-y']
    val_x = split_data['val-x']
    val_y = split_data['val-y']
    logistic.fit(train_x, train_y)

    train_pred = logistic.predict(train_x)
    train_prob = logistic.predict_proba(train_x)[:, 1]
    val_pred = logistic.predict(val_x)
    val_prob = logistic.predict_proba(val_x)[:, 1]

    train_acc = accuracy_score(train_y, train_pred)
    train_auc = roc_auc_score(train_y, train_prob)
    val_acc = accuracy_score(val_y, val_pred)
    val_auc = roc_auc_score(val_y, val_prob)

    result = {
        'train-acc': train_acc,
        'train-auc': train_auc,
        'val-acc': val_acc,
        'val-auc': val_auc
    }

    return result


# 3e) evaluate k-fold using logistic regression
def eval_kfold(x, y, k, logistic):
    fold_assign = generate_kfold(x, y, k)
    train_acc = []
    train_auc = []
    val_acc = []
    val_auc = []

    for fold in range(k):
        train_x = x[fold_assign != fold]
        train_y = y[fold_assign != fold]
        val_x = x[fold_assign == fold]
        val_y = y[fold_assign == fold]

        logistic.fit(train_x, train_y)

        train_pred = logistic.predict(train_x)
        train_prob = logistic.predict_proba(train_x)[:, 1]
        val_pred = logistic.predict(val_x)
        val_prob = logistic.predict_proba(val_x)[:, 1]

        train_acc.append(accuracy_score(train_y, train_pred))
        train_auc.append(roc_auc_score(train_y, train_prob))
        val_acc.append(accuracy_score(val_y, val_pred))
        val_auc.append(roc_auc_score(val_y, val_prob))

    mean_train_acc = np.mean(train_acc)
    mean_train_auc = np.mean(train_auc)
    mean_val_acc = np.mean(val_acc)
    mean_val_auc = np.mean(val_auc)

    result = {
        'train-acc': mean_train_acc,
        'train-auc': mean_train_auc,
        'val-acc': mean_val_acc,
        'val-auc': mean_val_auc
    }

    return result


def eval_mccv(x, y, valsize, s, logistic):
    train_acc = []
    train_auc = []
    val_acc = []
    val_auc = []
    for i in range(s):
        s_result = eval_holdout(x, y, valsize, logistic)
        train_acc.append(s_result['train-acc'])
        train_auc.append(s_result['train-auc'])
        val_acc.append(s_result['val-acc'])
        val_auc.append(s_result['val-auc'])

    mean_train_acc = np.mean(train_acc)
    mean_train_auc = np.mean(train_auc)
    mean_val_acc = np.mean(val_acc)
    mean_val_auc = np.mean(val_auc)

    result = {
        'train-acc': mean_train_acc,
        'train-auc': mean_train_auc,
        'val-acc': mean_val_acc,
        'val-auc': mean_val_auc
    }

    return result

