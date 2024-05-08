import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, mean_squared_error
import matplotlib.pyplot as plt
import time
import csv
import q2
import sgb


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


def feature_selection(x, y):
    corr_matrix = compute_correlation(x, corrtype='spearman')
    corr_df = pd.DataFrame(corr_matrix)
    # print(corr_df)
    upper = corr_df.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    drop = [column for column in upper.columns if any(abs(upper[column]) > 0.7)]
    # print(x)
    print(drop)
    new_x = pd.DataFrame(x).drop(columns=pd.DataFrame(x).columns[drop], axis=1).to_numpy()
    # print(new_x.shape)

    idx = rank_correlation(new_x, y)
    print('rank_correlation')
    print(f'ranking: {idx}')
    new_x = pd.DataFrame(new_x)[:][idx[:10]].to_numpy()

    return new_x, y


def preprocess_2b(df):
    x = df.drop('class', axis=1)
    y = df['class']
    # print(x.shape)
    # term
    le = LabelEncoder()
    le.fit(x['term'])
    # print(le.classes_)
    for label in le.classes_:
        x['term'].replace(to_replace=label, value=int(label[1:3]), inplace=True)

    # emp_length
    le = LabelEncoder()
    le.fit(x['emp_length'])
    # print(le.classes_)
    for label in le.classes_:
        if pd.isna(label):
            replace = -1
        elif label[0] == '<':
            replace = 0
        elif label[1] == '0':
            replace = 10
        else:
            replace = label[0]
        x['emp_length'].replace(to_replace=label, value=replace, inplace=True)

    # earliest_cr_line
    x['earliest_cr_line_year'] = pd.to_datetime(x['earliest_cr_line'], format="%b-%y").dt.strftime('%Y')
    # print(x['earliest_cr_line_year'])
    x = x.drop('earliest_cr_line', axis=1)

    # grade
    # home_ownership
    # verification_status
    # purpose
    for column in ['grade', 'home_ownership', 'verification_status', 'purpose']:
        le = LabelEncoder()
        x[column] = le.fit_transform(x[column])
        # print(le.classes_)

    x, y = feature_selection(x, y)

    # normalization
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    return x, y


def Q2():
    # Read in data
    df = pd.read_csv('loan_default.csv')
    # preprocess
    x, y = preprocess_2b(df)
    # 2a partition
    train_x, train_y, test_x, test_y = partition_2a(x, y, 0.2)

    print('2a')
    print('train_x size: ', train_x.shape)
    print('test_x size: ', test_x.shape)

    print('-----------------------------------------------------------------------------------------------------------')
    # 2d
    print('2d')
    hiddenparams = [(16,), (32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)]
    actparams = ['logistic', 'tanh', 'relu']
    alphaparams = [0.0001, 0.001, 0.01, 0.1, 1, 10]
    opt_params = q2.tune_nn(train_x, train_y, hiddenparams=hiddenparams, actparams=actparams, alphaparams=alphaparams)
    results = opt_params['results']
    opt_score = 0
    for i in range(len(results['mean_test_score'])):
        if results['rank_test_score'][i] == 1:
            opt_score = results['mean_test_score'][i]
            break
    print(f'opt_hidden: {opt_params["best-hidden"]}')
    print(f'opt_activation: {opt_params["best-activation"]}')
    print(f'opt_alpha: {opt_params["best-alpha"]}')
    print(f'opt_score: {opt_score}')
    print('-----------------------------------------------------------------------------------------------------------')

    # 2e
    print('2e')
    max_depth = 4
    min_sample = 100
    hidden = (64, 32)
    activation = 'logistic'
    alpha = 0.0001
    mlp = MLPClassifier(hidden_layer_sizes=hidden, activation=activation, alpha=alpha)
    tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_sample)

    start = time.time()
    mlp.fit(train_x, train_y)
    mlp_train_time = time.time() - start
    start = time.time()
    tree.fit(train_x, train_y)
    tree_train_time = time.time() - start

    mlp_pred = mlp.predict(test_x)
    tree_pred = tree.predict(test_x)

    mlp_auc = roc_auc_score(y_true=test_y, y_score=mlp_pred)
    mlp_f1 = f1_score(y_true=test_y, y_pred=mlp_pred)
    mlp_f2 = fbeta_score(y_true=test_y, y_pred=mlp_pred, beta=2)

    tree_auc = roc_auc_score(y_true=test_y, y_score=tree_pred)
    tree_f1 = f1_score(y_true=test_y, y_pred=tree_pred)
    tree_f2 = fbeta_score(y_true=test_y, y_pred=tree_pred, beta=2)

    mlp_result = {
        'model': 'MLP',
        'AUC': mlp_auc,
        'F1': mlp_f1,
        'F2': mlp_f2,
        'train_time': mlp_train_time
    }
    tree_result = {
        'model': 'Decision Tree',
        'AUC': tree_auc,
        'F1': tree_f1,
        'F2': tree_f2,
        'train_time': tree_train_time
    }
    print('MLP result: ')
    print(mlp_result)
    print('DT result: ')
    print(tree_result)

    columns = ['model', 'AUC', 'F1', 'F2', 'train_time']
    with open('q2e_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([mlp_result, tree_result])

    print('-----------------------------------------------------------------------------------------------------------')


def Q3():
    # read in data
    train_x = pd.read_csv('energydata\energy_train.csv')
    val_x = pd.read_csv('energydata\energy_val.csv')
    test_x = pd.read_csv('energydata\energy_test.csv')

    # create true values y
    train_y = train_x['Appliances'].to_numpy()
    val_y = val_x['Appliances'].to_numpy()
    test_y = test_x['Appliances'].to_numpy()

    # drop irrelevant features
    train_x = train_x.drop(['date', 'Appliances'], axis=1).to_numpy()
    val_x = val_x.drop(['date', 'Appliances'], axis=1).to_numpy()
    test_x = test_x.drop(['date', 'Appliances'], axis=1).to_numpy()

    # 3d
    print('3d')
    nu_search = [0, 0.1, 0.5, 0.01, 0.001]
    n_iter_search = [1, 5, 10, 25, 50]
    nu_plot = []
    n_iter_plot = []
    rmse_plot = []
    opt_rmse = 99999999
    opt_nu = -1,
    opt_n_iter = -1
    for nu in nu_search:
        for n_iter in n_iter_search:
            model = sgb.SGTB(nIter=n_iter, q=1, nu=nu)
            model.fit(train_x, train_y)
            pred_y = model.predict(val_x)
            rmse = mean_squared_error(val_y, pred_y, squared=False)

            nu_plot.append(nu)
            n_iter_plot.append(n_iter)
            rmse_plot.append(rmse)

            if rmse < opt_rmse:
                opt_rmse = rmse
                opt_nu = nu
                opt_n_iter = n_iter

    # plot 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(nu_plot, n_iter_plot, rmse_plot)
    ax.set_xlabel('nu')
    ax.set_ylabel('n_iter')
    ax.set_zlabel('RMSE')
    ax.set_title('SGTB performance with different nu and i_iter')

    plt.show()

    print(f'opt_nu: {opt_nu}')
    print(f'opt_n_iter: {opt_n_iter}')
    print(f'opt_rmse: {opt_rmse}')
    print('-----------------------------------------------------------------------------------------------------------')

    # 3f
    print('3f')
    q_lst = [0.6, 0.7, 0.8, 0.9]
    nu_lst = [0.1, 0.5, 0.01, 0.001]
    nIter_lst = [1, 5, 10, 25, 50]
    result = sgb.tune_sgtb(x=train_x, y=train_y, lst_nIter=nIter_lst, lst_nu=nu_lst, lst_q=q_lst, md=3)
    opt_nIter = result['best-nIter']
    opt_nu = result['best-nu']
    opt_q = result['best-q']
    results = []
    for i in range(len(result['results']['params'])):
        temp = result['results']['params'][i]
        temp['RMSE'] = abs(result['results']['mean_test_score'][i])
        results.append(temp)
    columns = ['nIter', 'nu', 'q', 'RMSE']
    with open('q3f_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(results)

    print(f'best_nIter: {opt_nIter}')
    print(f'best_nu: {opt_nu}')
    print(f'best_q: {opt_q}')
    print('-----------------------------------------------------------------------------------------------------------')

    # 3g
    print('3g')
    nIter = 25
    nu = 0.1
    q = 0.9

    train_x = np.concatenate((train_x, val_x))
    train_y = np.concatenate((train_y, val_y))

    model = sgb.SGTB(nIter=nIter, q=q, nu=nu, md=3)

    start = time.time()
    model.fit(train_x, train_y)
    sgd_train_time = time.time() - start
    pred_y = model.predict(test_x)
    rmse = mean_squared_error(test_y, pred_y, squared=False)
    print('SGTB: ')
    print(f'RMSE: {rmse}')
    print(f'train_time: {sgd_train_time}')
    print('-----------------------------------------------------------------------------------------------------------')

    # 3h
    print('3h')
    nIter = 10
    nu = 0.1
    q = 0.9

    train_x = np.concatenate((train_x, val_x))
    train_y = np.concatenate((train_y, val_y))

    model = sgb.SGTB(nIter=nIter, q=q, nu=nu, md=3)

    start = time.time()
    model.fit(train_x, train_y)
    sgd_train_time = time.time() - start
    pred_y = model.predict(test_x)
    rmse = mean_squared_error(test_y, pred_y, squared=False)
    print('SGTB: ')
    print(f'RMSE: {rmse}')
    print(f'train_time: {sgd_train_time}')

    nIter = 10
    nu = 0.1
    model = sgb.SGTB(nIter=nIter, q=1, nu=nu, md=3)

    start = time.time()
    model.fit(train_x, train_y)
    gtb_train_time = time.time() - start
    pred_y = model.predict(test_x)
    rmse = mean_squared_error(test_y, pred_y, squared=False)
    print('GTB')
    print(f'RMSE: {rmse}')
    print(f'train_time: {gtb_train_time}')

def main():
    # Q2
    Q2()

    # Q3
    Q3()


if __name__ == '__main__':
    main()
