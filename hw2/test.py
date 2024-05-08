import numpy as np
import pandas as pd
import q2
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

import q3


def main():
    # preparation
    # load data
    # assuming the data are in the current working directory
    file_path = ''
    train_file = file_path + 'spam.train.dat'
    test_file = file_path+'spam.test.dat'
    train = np.genfromtxt(train_file)
    test = np.genfromtxt(test_file)
    # print(train.shape)
    # print(test)
    # split data into features and labels
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]
    # print(train_x.shape)
    # print(train_y.shape)
    # print(train_y)

    # ------------------------------------------------------------------------------------
    # 2c) report naive bayes on each of the preprocessing methods
    columns = ['train-acc', 'train-auc', 'test-acc', 'test-auc']
    # preprocess data
    nothing_train_x, nothing_test_x = q2.do_nothing(train_x, test_x)
    std_train_x, std_test_x = q2.do_std(train_x, test_x)
    log_train_x, log_test_x = q2.do_log(train_x, test_x)
    bin_train_x, bin_test_x = q2.do_bin(train_x, test_x)

    # do_nothing
    nothing_nb = q2.eval_nb(nothing_train_x, train_y, nothing_test_x, test_y)
    nothing_nb_result = {'method': 'nothing'}
    nothing_nb_result.update({key: nothing_nb[key] for key in columns})
    print(nothing_nb_result)
    # do_std
    std_nb = q2.eval_nb(std_train_x, train_y, std_test_x, test_y)
    std_nb_result = {'method': 'std'}
    std_nb_result.update({key: std_nb[key] for key in columns})
    print(std_nb_result)
    # do_log
    log_nb = q2.eval_nb(log_train_x, train_y, log_test_x, test_y)
    log_nb_result = {'method': 'log'}
    log_nb_result.update({key: log_nb[key] for key in columns})
    print(log_nb_result)
    # do_bin
    bin_nb = q2.eval_nb(bin_train_x, train_y, bin_test_x, test_y)
    bin_nb_result = {'method': 'bin'}
    bin_nb_result.update({key: bin_nb[key] for key in columns})
    print(bin_nb_result)

    columns = ['method', 'train-acc', 'train-auc', 'test-acc', 'test-auc']
    with open('q2_nb_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([nothing_nb_result, std_nb_result, log_nb_result, bin_nb_result])

    print('---------------------------------------------------------------------------')
    # 2e) report logistic regression on each of the preprocessing methods
    columns = ['train-acc', 'train-auc', 'test-acc', 'test-auc']
    # do_nothing
    nothing_lr = q2.eval_lr(nothing_train_x, train_y, nothing_test_x, test_y)
    nothing_lr_result = {'method': 'nothing'}
    nothing_lr_result.update({key: nothing_lr[key] for key in columns})
    print(nothing_lr_result)
    # do_std
    std_lr = q2.eval_lr(std_train_x, train_y, std_test_x, test_y)
    std_lr_result = {'method': 'std'}
    std_lr_result.update({key: std_lr[key] for key in columns})
    print(std_lr_result)
    # do_log
    log_lr = q2.eval_lr(log_train_x, train_y, log_test_x, test_y)
    log_lr_result = {'method': 'log'}
    log_lr_result.update({key: log_lr[key] for key in columns})
    print(log_lr_result)
    # do_bin
    bin_lr = q2.eval_lr(bin_train_x, train_y, bin_test_x, test_y)
    bin_lr_result = {'method': 'bin'}
    bin_lr_result.update({key: bin_lr[key] for key in columns})
    print(bin_lr_result)

    columns = ['method', 'train-acc', 'train-auc', 'test-acc', 'test-auc']
    with open('q2_lr_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([nothing_lr_result, std_lr_result, log_lr_result, bin_lr_result])

    print('---------------------------------------------------------------------------')
    # 2f) plot ROC curves for the test data
    # Naive Bayes:
    nb_nothing_fpr, nb_nothing_tpr, _ = roc_curve(test_y, nothing_nb['test-prob'])
    nb_std_fpr, nb_std_tpr, _ = roc_curve(test_y, std_nb['test-prob'])
    nb_log_fpr, nb_log_tpr, _ = roc_curve(test_y, log_nb['test-prob'])
    nb_bin_fpr, nb_bin_tpr, _ = roc_curve(test_y, bin_nb['test-prob'])

    plt.figure(figsize=(12, 6))
    plt.title('ROC Curves for Naive Bayes')
    plt.plot(nb_nothing_fpr, nb_nothing_tpr, label='nb_nothing')
    plt.plot(nb_std_fpr, nb_std_tpr, label='nb_std')
    plt.plot(nb_log_fpr, nb_log_tpr, label='nb_log')
    plt.plot(nb_bin_fpr, nb_bin_tpr, label='nb_bin')
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

    plt.clf()
    # Logistic Regression:
    lr_nothing_fpr, lr_nothing_tpr, _ = roc_curve(test_y, nothing_lr['test-prob'])
    lr_std_fpr, lr_std_tpr, _ = roc_curve(test_y, std_lr['test-prob'])
    lr_log_fpr, lr_log_tpr, _ = roc_curve(test_y, log_lr['test-prob'])
    lr_bin_fpr, lr_bin_tpr, _ = roc_curve(test_y, bin_lr['test-prob'])

    plt.figure(figsize=(12, 6))
    plt.title('ROC Curves for Standard Logistic Regression')
    plt.plot(lr_nothing_fpr, lr_nothing_tpr, label='lr_nothing')
    plt.plot(lr_std_fpr, lr_std_tpr, label='lr_std')
    plt.plot(lr_log_fpr, lr_log_tpr, label='lr_log')
    plt.plot(lr_bin_fpr, lr_bin_tpr, label='lr_bin')
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

    plt.clf()

    # best from naive bayes and logistic regression -- both log
    plt.figure(figsize=(12, 6))
    plt.title('ROC Curves of Best Preprocessing Method (log) for Naive Bayes and Logistic Regression')
    plt.plot(nb_log_fpr, nb_log_tpr, label='nb_log')
    plt.plot(lr_log_fpr, lr_log_tpr, label='lr_log')
    plt.legend()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid()
    plt.show()

    plt.clf()
    # ---------------------------------------------------------------------------------------
    # 3g) testing holdout on ridge and LASSO
    alpha_space = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    valsize_space = [0.2, 0.5, 0.8]
    columns = ['alpha', 'ridge-acc', 'ridge-auc', 'lasso-acc', 'lasso-auc']

    for valsize in valsize_space:
        result_row = []

        for alpha in alpha_space:
            ridge = LogisticRegression(penalty='l2', C=1/alpha, solver='liblinear')
            lasso = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear')
            ridge_result = q3.eval_holdout(log_train_x, train_y, valsize, ridge)
            lasso_result = q3.eval_holdout(log_train_x, train_y, valsize, lasso)
            result_row.append({
                'alpha': alpha,
                'ridge-acc': ridge_result['val-acc'],
                'ridge-auc': ridge_result['val-auc'],
                'lasso-acc': lasso_result['val-acc'],
                'lasso-auc': lasso_result['val-auc']
            })
        print(result_row)
        with open(f'q3_holdout_{valsize}.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result_row)

    print('--------------------------------------------------------------------------')
    # 3h) testing k-fold on ridge and LASSO
    k_space = [2, 5, 10]
    for k in k_space:
        result_row = []

        for alpha in alpha_space:
            ridge = LogisticRegression(penalty='l2', C=1/alpha, solver='liblinear')
            lasso = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear')
            ridge_result = q3.eval_kfold(log_train_x, train_y, k, ridge)
            lasso_result = q3.eval_kfold(log_train_x, train_y, k, lasso)
            result_row.append({
                'alpha': alpha,
                'ridge-acc': ridge_result['val-acc'],
                'ridge-auc': ridge_result['val-auc'],
                'lasso-acc': lasso_result['val-acc'],
                'lasso-auc': lasso_result['val-auc']
            })
        print(result_row)
        with open(f'q3_kfold_{k}.csv', 'w') as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(result_row)

    print('--------------------------------------------------------------------------')
    # 3j) test Monte Carlo on Ridge and LASSO
    columns = ['s', 'valsize', 'alpha', 'ridge-acc', 'ridge-auc', 'lasso-acc', 'lasso-auc']
    s_space = [5, 10]
    result_row = []
    for s in s_space:
        for valsize in valsize_space:
            for alpha in alpha_space:
                ridge = LogisticRegression(penalty='l2', C=1 / alpha, solver='liblinear')
                lasso = LogisticRegression(penalty='l1', C=1 / alpha, solver='liblinear')
                ridge_result = q3.eval_mccv(log_train_x, train_y, valsize, s, ridge)
                lasso_result = q3.eval_mccv(log_train_x, train_y, valsize, s, lasso)
                result_row.append({
                    's': s,
                    'valsize': valsize,
                    'alpha': alpha,
                    'ridge-acc': ridge_result['val-acc'],
                    'ridge-auc': ridge_result['val-auc'],
                    'lasso-acc': lasso_result['val-acc'],
                    'lasso-auc': lasso_result['val-auc']
                })
    print(result_row)
    with open('q3_mccv.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_row)

    print('--------------------------------------------------------------------------')
    # 3k) Ridge and LASSO with optimal alphas
    ridge_alpha = [0.01, 0.1, 1, 10]
    lasso_alpha = [0.01, 0.1, 1, 10]
    result_row = []
    columns = ['alpha', 'ridge-acc', 'ridge-auc', 'lasso-acc', 'lasso-auc']
    for alpha in ridge_alpha:
        ridge = LogisticRegression(penalty='l2', C=1 / alpha, solver='liblinear')
        lasso = LogisticRegression(penalty='l1', C=1 / alpha, solver='liblinear')
        ridge.fit(log_train_x, train_y)
        lasso.fit(log_train_x, train_y)

        ridge_pred = ridge.predict(log_test_x)
        ridge_prob = ridge.predict_proba(log_test_x)[:, 1]
        lasso_pred = lasso.predict(log_test_x)
        lasso_prob = lasso.predict_proba(log_test_x)[:, 1]

        ridge_acc = accuracy_score(test_y, ridge_pred)
        ridge_auc = roc_auc_score(test_y, ridge_prob)
        lasso_acc = accuracy_score(test_y, lasso_pred)
        lasso_auc = roc_auc_score(test_y, lasso_prob)

        result_row.append({
            'alpha': alpha,
            'ridge-acc': ridge_acc,
            'ridge-auc': ridge_auc,
            'lasso-acc': lasso_acc,
            'lasso-auc': lasso_auc
        })

    print(result_row)

    with open('q3_optimal.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_row)


if __name__ == '__main__':
    main()
