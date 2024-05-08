import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def preprocess_data(trainx, valx, testx):
    # preprocess data using StandardScaler
    # n_feature = trainx.shape[1]
    #
    # new_train = np.zeros_like(trainx, dtype=float)
    # new_val = np.zeros_like(valx, dtype=float)
    # new_test = np.zeros_like(testx, dtype=float)
    #
    # # Scale each feature separately
    # for i in range(n_feature):
    #     scaler = StandardScaler()
    #     train = trainx[:, i]
    #     val = valx[:, i]
    #     test = testx[:, i]
    #
    #     scaler.fit(train.reshape(-1, 1))
    #
    #     train = scaler.transform(train.reshape(-1, 1))
    #     val = scaler.transform(val.reshape(-1, 1))
    #     test = scaler.transform(test.reshape(-1, 1))
    #
    #     new_train[:, i] = train.flatten()
    #     new_val[:, i] = val.flatten()
    #     new_test[:, i] = test.flatten()

    scaler = StandardScaler()
    new_train = scaler.fit_transform(trainx)
    new_val = scaler.transform(valx)
    new_test = scaler.transform(testx)


    return new_train, new_val, new_test


# Linear Regression model trained only on train set
def eval_linear1(trainx, trainy, valx, valy, testx, testy):
    model = linear_model.LinearRegression()
    # train model on train set only
    model.fit(trainx, trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result


# Linear Regression model trained on train and val sets
def eval_linear2(trainx, trainy, valx, valy, testx, testy):
    model = linear_model.LinearRegression()
    total_trainx = np.concatenate((trainx, valx), axis=0)
    # print(trainx.shape)
    # print(total_trainx.shape)
    total_trainy = np.concatenate((trainy, valy))
    # print(trainy.shape)
    # print(total_trainy.shape)
    model.fit(total_trainx, total_trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result


# ridge regression model trained only on train set
def eval_ridge1(trainx, trainy, valx, valy, testx, testy, alpha):
    model = linear_model.Ridge(alpha=alpha)

    model.fit(trainx, trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result


# lasso regression model trained only on train set
def eval_lasso1(trainx, trainy, valx, valy, testx, testy, alpha):
    model = linear_model.Lasso(alpha=alpha)

    model.fit(trainx, trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result


def eval_ridge2(trainx, trainy, valx, valy, testx, testy, alpha):
    model = linear_model.Ridge(alpha=alpha)
    total_trainx = np.concatenate((trainx, valx), axis=0)
    # print(trainx.shape)
    # print(total_trainx.shape)
    total_trainy = np.concatenate((trainy, valy))
    # print(trainy.shape)
    # print(total_trainy.shape)
    model.fit(total_trainx, total_trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result



def eval_lasso2(trainx, trainy, valx, valy, testx, testy, alpha):
    model = linear_model.Lasso(alpha=alpha)
    total_trainx = np.concatenate((trainx, valx), axis=0)
    # print(trainx.shape)
    # print(total_trainx.shape)
    total_trainy = np.concatenate((trainy, valy))
    # print(trainy.shape)
    # print(total_trainy.shape)
    model.fit(total_trainx, total_trainy)

    # test model on train, val, and test sets separately
    train_y_pred = model.predict(trainx)
    val_y_pred = model.predict(valx)
    test_y_pred = model.predict(testx)

    # calculate metrics
    train_rmse = mean_squared_error(y_true=trainy, y_pred=train_y_pred, squared=False)
    val_rmse = mean_squared_error(y_true=valy, y_pred=val_y_pred, squared=False)
    test_rmse = mean_squared_error(y_true=testy, y_pred=test_y_pred, squared=False)

    train_r2 = r2_score(y_true=trainy, y_pred=train_y_pred)
    val_r2 = r2_score(y_true=valy, y_pred=val_y_pred)
    test_r2 = r2_score(y_true=testy, y_pred=test_y_pred)

    result = {
        'train-rmse': train_rmse,
        'train-r2': train_r2,
        'val-rmse': val_rmse,
        'val-r2': val_r2,
        'test-rmse': test_rmse,
        'test-r2': test_r2
    }

    return result


