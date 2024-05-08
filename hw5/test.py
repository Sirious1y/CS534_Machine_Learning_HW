import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import q2, q3


def partition_2a(x, y, test_size):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
    return train_x, train_y, test_x, test_y


def compute_correlation(x, corrtype):
    df = pd.DataFrame(x)
    corr = df.corr(method=corrtype)
    return corr.to_numpy()


def rank_correlation(x, y):
    x = np.transpose(x)
    corrs = []
    for column in x:
        covar = np.cov(column.astype(float), y)
        # print(covar)
        corr = covar[0, 1]/np.sqrt(covar[0, 0] * covar[1, 1])
        corrs.append(corr)
    corrs = np.abs(corrs)
    # print(corrs)
    sorted_idx = np.argsort(corrs)[::-1]
    return sorted_idx


def feature_selection_2d(x, y, test_x, test_y):
    corr_matrix = compute_correlation(x, corrtype='spearman')
    corr_df = pd.DataFrame(corr_matrix)
    # print(corr_df)
    upper = corr_df.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    drop = [column for column in upper.columns if any(abs(upper[column]) > 0.7)]
    # print(x)
    # print(drop)
    new_x = pd.DataFrame(x).drop(columns=pd.DataFrame(x).columns[drop], axis=1).to_numpy()
    new_test_x = pd.DataFrame(test_x).drop(columns=pd.DataFrame(x).columns[drop], axis=1).to_numpy()
    # print(new_x.shape)

    idx = rank_correlation(new_x, y)
    # print('rank_correlation')
    # print(f'ranking: {idx}')
    new_x = pd.DataFrame(new_x)[:][idx[:10]].to_numpy()
    new_test_x = pd.DataFrame(new_test_x)[:][idx[:10]].to_numpy()

    return new_x, y, new_test_x, test_y


def Q1():
    # 1b
    frac = 0.01
    d = np.array([1, 10, 100, 1000])
    f = 1 - (1 - frac) ** d

    fig, ax = plt.subplots()
    ax.plot(d, f)
    ax.set_xscale('log')
    ax.set_xlabel('d')
    ax.set_ylabel('f')
    ax.set_title('f vs d with $\epsilon/a=0.01$')

    plt.show()


def Q2():
    # Read in data
    df = pd.read_csv('loan_default.csv')
    x = df.drop('class', axis=1)
    y = df['class'].to_numpy()
    features = df.columns

    # 2b preprocess
    x = q2.preprocess(x)

    # 2a partition
    train_x, train_y, test_x, test_y = partition_2a(x, y, test_size=0.2)
    train = np.column_stack([train_x, train_y])
    test = np.column_stack([test_x, test_y])

    np.savetxt("loan_train.csv", train, delimiter=',')
    np.savetxt("loan_test.csv", test, delimiter=',')

    # 2d feature selection
    fs_train_x, fs_train_y, fs_test_x, fs_test_y = feature_selection_2d(train_x, train_y, test_x, test_y)
    fs_train = np.column_stack([fs_train_x, fs_train_y])
    fs_test = np.column_stack([fs_test_x, fs_test_y])

    np.savetxt("loan_fs_train.csv", fs_train, delimiter=',')
    np.savetxt("loan_fs_test.csv", fs_test, delimiter=',')
    # ------------------------------------------------------------------------------------------------------

    print('2g')
    pca_num_components, pca_components, pca_train_x, pca_test_x = q2.run_pca(train_x, test_x)

    print(f'num to capture 95% variance: {pca_num_components}')
    # print('First 3 principal components')
    # print(pca_components[:3])
    print('Top 3 features that contribute the most for the first 3 principal components')
    for i in range(3):
        idx = np.argpartition(pca_components[i], -3)[-3:]
        print(f'Principal component {i+1}: {idx}, {features[idx]}')
    print('------------------------------------------------------------------------------------------------------')

    # 2h
    pca_train = np.column_stack([pca_train_x, train_y])
    pca_test = np.column_stack([pca_test_x, test_y])
    np.savetxt("loan_pca_train.csv", pca_train, delimiter=',')
    np.savetxt("loan_pca_test.csv", pca_test, delimiter=',')

    print('2j')
    k_lst = list(range(1, 27))
    recon_errs = []
    for k in range(1, 27):
        nmf_recon_err, nmf_components, nmf_train_x, nmf_test_x = q2.run_nmf(train_x, test_x, k=k)
        recon_errs.append(nmf_recon_err)

    plt.plot(k_lst, recon_errs)
    plt.xlabel('number of components')
    plt.ylabel('reconstruction error')
    plt.title('recon_err vs. k')
    plt.show()

    print(f'optimal number of components: {k_lst[np.argmin(recon_errs)]}')

    nmf_recon_err, nmf_components, nmf_train_x, nmf_test_x = q2.run_nmf(train_x, test_x, k=20)

    # print('First 3 factor components')
    # print(nmf_components[:3])
    print(nmf_components.shape)
    print('Top 3 features that contribute the most for the first 3 factors')
    for i in range(3):
        idx = np.argpartition(nmf_components[i], -3)[-3:]
        print(f'Factor component {i+1}: {idx}, {features[idx]}')

    print('------------------------------------------------------------------------------------------------------')

    # 2k
    nmf_train = np.column_stack([nmf_train_x, train_y])
    nmf_test = np.column_stack([nmf_test_x, test_y])
    np.savetxt("loan_nmf_train.csv", nmf_train, delimiter=',')
    np.savetxt("loan_nmf_test.csv", nmf_test, delimiter=',')


def Q3():
    # only preprocessed
    train = np.genfromtxt('loan_train.csv', delimiter=',')
    test = np.genfromtxt('loan_test.csv', delimiter=',')
    train_x = train[:, :-1]
    train_y = train[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]
    scaler = StandardScaler()
    train_x = scaler.fit_transform(train_x)
    test_x = scaler.transform(test_x)

    # feature selection
    train = np.genfromtxt('loan_fs_train.csv', delimiter=',')
    test = np.genfromtxt('loan_fs_test.csv', delimiter=',')
    fs_train_x = train[:, :-1]
    fs_train_y = train[:, -1]
    fs_test_x = test[:, :-1]
    fs_test_y = test[:, -1]
    scaler = StandardScaler()
    fs_train_x = scaler.fit_transform(fs_train_x)
    fs_test_x = scaler.transform(fs_test_x)

    # pca
    train = np.genfromtxt('loan_pca_train.csv', delimiter=',')
    test = np.genfromtxt('loan_pca_test.csv', delimiter=',')
    pca_train_x = train[:, :-1]
    pca_train_y = train[:, -1]
    pca_test_x = test[:, :-1]
    pca_test_y = test[:, -1]

    # nmf
    train = np.genfromtxt('loan_nmf_train.csv', delimiter=',')
    test = np.genfromtxt('loan_nmf_test.csv', delimiter=',')
    nmf_train_x = train[:, :-1]
    nmf_train_y = train[:, -1]
    nmf_test_x = test[:, :-1]
    nmf_test_y = test[:, -1]
    scaler = MinMaxScaler()
    nmf_train_x = scaler.fit_transform(nmf_train_x)
    nmf_test_x = scaler.transform(nmf_test_x)

    columns = ['model', 'preprocess', 'train-auc', 'train-f1', 'train-f2', 'val-auc', 'val-f1', 'val-f2', 'test-auc',
               'test-f1', 'test-f2']

    # logr
    print('logr')
    base_logr_result = {
        'model': 'logr',
        'preprocess': 'baseline'
    }
    fs_logr_result = {
        'model': 'logr',
        'preprocess': 'fs'
    }
    pca_logr_result = {
        'model': 'logr',
        'preprocess': 'pca'
    }
    nmf_logr_result = {
        'model': 'logr',
        'preprocess': 'nmf'
    }
    base_logr_result.update(q3.build_logr(train_x, test_x, train_y, test_y))
    base_logr_result.pop('params', None)
    fs_logr_result.update(q3.build_logr(fs_train_x, fs_test_x, fs_train_y, fs_test_y))
    fs_logr_result.pop('params', None)
    pca_logr_result.update(q3.build_logr(pca_train_x, pca_test_x, pca_train_y, pca_test_y))
    pca_logr_result.pop('params', None)
    nmf_logr_result.update(q3.build_logr(nmf_train_x, nmf_test_x, nmf_train_y, nmf_test_y))
    nmf_logr_result.pop('params', None)
    # print(base_logr_result)
    with open('q3_logr.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base_logr_result, fs_logr_result, pca_logr_result, nmf_logr_result])
    print('------------------------------------------------------------------------------------------')

    # dt
    print('dt: ')
    base_dt_result = {
        'model': 'dt',
        'preprocess': 'baseline'
    }
    fs_dt_result = {
        'model': 'dt',
        'preprocess': 'fs'
    }
    pca_dt_result = {
        'model': 'dt',
        'preprocess': 'pca'
    }
    nmf_dt_result = {
        'model': 'dt',
        'preprocess': 'nmf'
    }
    print('baseline: ')
    base_dt_result.update(q3.build_dt(train_x, test_x, train_y, test_y))
    base_dt_result.pop('params', None)
    print('fs: ')
    fs_dt_result.update(q3.build_dt(fs_train_x, fs_test_x, fs_train_y, fs_test_y))
    fs_dt_result.pop('params', None)
    print('pca: ')
    pca_dt_result.update(q3.build_dt(pca_train_x, pca_test_x, pca_train_y, pca_test_y))
    pca_dt_result.pop('params', None)
    print('nmf: ')
    nmf_dt_result.update(q3.build_dt(nmf_train_x, nmf_test_x, nmf_train_y, nmf_test_y))
    nmf_dt_result.pop('params', None)
    with open('q3_dt.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base_dt_result, fs_dt_result, pca_dt_result, nmf_dt_result])
    print('------------------------------------------------------------------------------------------')

    # rf
    print('rf: ')
    base_rf_result = {
        'model': 'rf',
        'preprocess': 'baseline'
    }
    fs_rf_result = {
        'model': 'rf',
        'preprocess': 'fs'
    }
    pca_rf_result = {
        'model': 'rf',
        'preprocess': 'pca'
    }
    nmf_rf_result = {
        'model': 'rf',
        'preprocess': 'nmf'
    }
    print('baseline: ')
    base_rf_result.update(q3.build_rf(train_x, test_x, train_y, test_y))
    base_rf_result.pop('params', None)
    print('fs: ')
    fs_rf_result.update(q3.build_rf(fs_train_x, fs_test_x, fs_train_y, fs_test_y))
    fs_rf_result.pop('params', None)
    print('pca: ')
    pca_rf_result.update(q3.build_rf(pca_train_x, pca_test_x, pca_train_y, pca_test_y))
    pca_rf_result.pop('params', None)
    print('nmf: ')
    nmf_rf_result.update(q3.build_rf(nmf_train_x, nmf_test_x, nmf_train_y, nmf_test_y))
    nmf_rf_result.pop('params', None)
    with open('q3_rf.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base_rf_result, fs_rf_result, pca_rf_result, nmf_rf_result])
    print('------------------------------------------------------------------------------------------')

    # svm
    print('svm: ')
    base_svm_result = {
        'model': 'svm',
        'preprocess': 'baseline'
    }
    fs_svm_result = {
        'model': 'svm',
        'preprocess': 'fs'
    }
    pca_svm_result = {
        'model': 'svm',
        'preprocess': 'pca'
    }
    nmf_svm_result = {
        'model': 'svm',
        'preprocess': 'nmf'
    }
    print('baseline: ')
    base_svm_result.update(q3.build_svm(train_x, test_x, train_y, test_y))
    base_svm_result.pop('params', None)
    print('fs: ')
    fs_svm_result.update(q3.build_svm(fs_train_x, fs_test_x, fs_train_y, fs_test_y))
    fs_svm_result.pop('params', None)
    print('pca: ')
    pca_svm_result.update(q3.build_svm(pca_train_x, pca_test_x, pca_train_y, pca_test_y))
    pca_svm_result.pop('params', None)
    print('nmf: ')
    nmf_svm_result.update(q3.build_svm(nmf_train_x, nmf_test_x, nmf_train_y, nmf_test_y))
    nmf_svm_result.pop('params', None)
    with open('q3_svm.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base_svm_result, fs_svm_result, pca_svm_result, nmf_svm_result])
    print('------------------------------------------------------------------------------------------')

    # nn
    print('nn: ')
    base_nn_result = {
        'model': 'nn',
        'preprocess': 'baseline'
    }
    fs_nn_result = {
        'model': 'nn',
        'preprocess': 'fs'
    }
    pca_nn_result = {
        'model': 'nn',
        'preprocess': 'pca'
    }
    nmf_nn_result = {
        'model': 'nn',
        'preprocess': 'nmf'
    }
    print('baseline: ')
    base_nn_result.update(q3.build_nn(train_x, test_x, train_y, test_y))
    base_nn_result.pop('params', None)
    print('fs: ')
    fs_nn_result.update(q3.build_nn(fs_train_x, fs_test_x, fs_train_y, fs_test_y))
    fs_nn_result.pop('params', None)
    print('pca: ')
    pca_nn_result.update(q3.build_nn(pca_train_x, pca_test_x, pca_train_y, pca_test_y))
    pca_nn_result.pop('params', None)
    print('nmf: ')
    nmf_nn_result.update(q3.build_nn(nmf_train_x, nmf_test_x, nmf_train_y, nmf_test_y))
    nmf_nn_result.pop('params', None)
    with open('q3_nn.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([base_nn_result, fs_nn_result, pca_nn_result, nmf_nn_result])
    print('------------------------------------------------------------------------------------------')


def main():
    # Q1()
    Q2()
    # Q3()


if __name__ == '__main__':
    main()
