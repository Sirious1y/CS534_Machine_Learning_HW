import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF


# 2c
def preprocess(x):
    # term
    # print('term')
    if x['term'].dtype.name == 'object':
        le = LabelEncoder()
        le.fit(x['term'])
        # print(le.classes_)
        for label in le.classes_:
            x['term'].replace(to_replace=label, value=int(label[1:3]), inplace=True)

    # emp_length
    if x['emp_length'].dtype.name == 'object':
        # print('emp_length')
        le = LabelEncoder()
        le.fit(x['emp_length'])
        # print(le.classes_)
        for label in le.classes_:
            if pd.isna(label):
                replace = 0
            elif label[0] == '<':
                replace = 1
            elif label[1] == '0':
                replace = 11
            else:
                replace = int(label[0]) + 1
            x['emp_length'].replace(to_replace=label, value=replace, inplace=True)

    # earliest_cr_line
    if x['earliest_cr_line'].dtype.name == 'object':
        # print(x['earliest_cr_line'].dtype)
        # print(x['earliest_cr_line'].unique())
        x['earliest_cr_line'] = pd.to_datetime(x['earliest_cr_line'], format="%b-%y").dt.strftime('%Y')
        x['earliest_cr_line'] = x['earliest_cr_line'].astype(int)

    # grade
    # home_ownership
    # verification_status
    # purpose
    for column in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        if x[column].dtype.name == 'object':
            le = LabelEncoder()
            x[column] = le.fit_transform(x[column])
            # print(le.classes_)

    # # normalization
    # scaler = StandardScaler()
    # x = scaler.fit_transform(x)

    return x.to_numpy().astype(float)


def run_pca(train_x, test_x):
    # normalize
    scaler = StandardScaler()
    train_x_normalized = scaler.fit_transform(train_x)
    test_x_normalized = scaler.transform(test_x)

    pca = PCA()
    pca.fit(train_x_normalized)

    cumulative_var = np.cumsum(pca.explained_variance_ratio_)
    num_components = int(np.argmax(cumulative_var >= 0.95) + 1)

    pca = PCA(n_components=num_components)
    train_loadings = pca.fit_transform(train_x_normalized)
    test_loadings = pca.transform(test_x_normalized)

    return num_components, pca.components_, train_loadings, test_loadings


def run_nmf(train_x, test_x, k):
    nmf = NMF(n_components=k, init='random', max_iter=5000)
    train_loadings = nmf.fit_transform(train_x)
    test_loadings = nmf.transform(test_x)

    # print(nmf.components_.shape)
    # print(train_loadings.shape)
    return nmf.reconstruction_err_, nmf.components_, train_loadings, test_loadings
