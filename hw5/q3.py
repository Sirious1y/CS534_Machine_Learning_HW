import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score, make_scorer


# 3a
def build_logr(train_x, test_x, train_y, test_y):
    train_scores = {
        'auc': [],
        'f1': [],
        'f2': []
    }
    val_scores = {
        'auc': [],
        'f1': [],
        'f2': []
    }

    kf = KFold(n_splits=5, shuffle=True)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_x)):
        k_train_x = train_x[train_idx]
        k_train_y = train_y[train_idx]
        k_val_x = train_x[val_idx]
        k_val_y = train_y[val_idx]

        model = LogisticRegression(penalty=None, max_iter=2000)
        model.fit(k_train_x, k_train_y)

        k_train_pred = model.predict(k_train_x)
        train_scores['auc'].append(roc_auc_score(k_train_y, model.predict_proba(k_train_x)[:, 1]))
        train_scores['f1'].append(f1_score(k_train_y, k_train_pred))
        train_scores['f2'].append(fbeta_score(k_train_y, k_train_pred, beta=2))

        k_val_pred = model.predict(k_val_x)
        val_scores['auc'].append(roc_auc_score(k_val_y, model.predict_proba(k_val_x)[:, 1]))
        val_scores['f1'].append(f1_score(k_val_y, k_val_pred))
        val_scores['f2'].append(fbeta_score(k_val_y, k_val_pred, beta=2))

    train_auc = np.average(train_scores['auc'])
    train_f1 = np.average(train_scores['f1'])
    train_f2 = np.average(train_scores['f2'])

    val_auc = np.average(val_scores['auc'])
    val_f1 = np.average(val_scores['f1'])
    val_f2 = np.average(val_scores['f2'])

    model = LogisticRegression(penalty=None, max_iter=2000)
    model.fit(train_x, train_y)
    pred_y = model.predict(test_x)

    test_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    test_f1 = f1_score(test_y, pred_y)
    test_f2 = fbeta_score(test_y, pred_y, beta=2)

    result = {
        'train-auc': train_auc,
        'train-f1': train_f1,
        'train-f2': train_f2,
        'val-auc': val_auc,
        'val-f1': val_f1,
        'val-f2': val_f2,
        'test-auc': test_auc,
        'test-f1': test_f1,
        'test-f2': test_f2,
        'params': {}
    }
    return result


# 3b
def build_dt(train_x, test_x, train_y, test_y):
    params = {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        'min_samples_leaf': [1, 5, 10, 50, 100, 150, 200, 300, 500]
    }
    scoring = {
        'AUC': 'roc_auc',
        'F1': make_scorer(f1_score),
        'F2': make_scorer(fbeta_score, beta=2)
    }
    grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=params, scoring=scoring,
                               refit='AUC', return_train_score=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    opt_params = grid_search.best_params_
    grid_search_results = grid_search.cv_results_
    opt_idx = np.argmax(grid_search_results['mean_test_AUC'])
    # print(f'best_params: {opt_params}')

    train_auc = grid_search_results['mean_train_AUC'][opt_idx]
    train_f1 = grid_search_results['mean_train_F1'][opt_idx]
    train_f2 = grid_search_results['mean_train_F2'][opt_idx]
    val_auc = grid_search_results['mean_test_AUC'][opt_idx]
    val_f1 = grid_search_results['mean_test_F1'][opt_idx]
    val_f2 = grid_search_results['mean_test_F2'][opt_idx]

    # model = DecisionTreeClassifier(max_depth=opt_params['max_depth'], min_samples_leaf=opt_params['min_samples_leaf'])
    # model.fit(train_x, train_y)
    model = grid_search.best_estimator_
    pred_y = model.predict(test_x)

    test_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    test_f1 = f1_score(test_y, pred_y)
    test_f2 = fbeta_score(test_y, pred_y, beta=2)

    result = {
        'train-auc': train_auc,
        'train-f1': train_f1,
        'train-f2': train_f2,
        'val-auc': val_auc,
        'val-f1': val_f1,
        'val-f2': val_f2,
        'test-auc': test_auc,
        'test-f1': test_f1,
        'test-f2': test_f2,
        'params': params
    }
    return result


# 3c
def build_rf(train_x, test_x, train_y, test_y):
    params = {
        'n_estimators': [1, 3, 5, 10],
        'max_depth': [5, 10, 15, 20],
        'min_samples_leaf': [50, 100, 200, 300, 500]
    }
    scoring = {
        'AUC': 'roc_auc',
        'F1': make_scorer(f1_score),
        'F2': make_scorer(fbeta_score, beta=2)
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=params, scoring=scoring,
                               refit='AUC', return_train_score=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    opt_params = grid_search.best_params_
    grid_search_results = grid_search.cv_results_
    opt_idx = np.argmax(grid_search_results['mean_test_AUC'])
    # print(f'best_params: {opt_params}')

    train_auc = grid_search_results['mean_train_AUC'][opt_idx]
    train_f1 = grid_search_results['mean_train_F1'][opt_idx]
    train_f2 = grid_search_results['mean_train_F2'][opt_idx]
    val_auc = grid_search_results['mean_test_AUC'][opt_idx]
    val_f1 = grid_search_results['mean_test_F1'][opt_idx]
    val_f2 = grid_search_results['mean_test_F2'][opt_idx]

    model = grid_search.best_estimator_
    pred_y = model.predict(test_x)

    test_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    test_f1 = f1_score(test_y, pred_y)
    test_f2 = fbeta_score(test_y, pred_y, beta=2)

    result = {
        'train-auc': train_auc,
        'train-f1': train_f1,
        'train-f2': train_f2,
        'val-auc': val_auc,
        'val-f1': val_f1,
        'val-f2': val_f2,
        'test-auc': test_auc,
        'test-f1': test_f1,
        'test-f2': test_f2,
        'params': params
    }
    return result


# 3d
def build_svm(train_x, test_x, train_y, test_y):
    params = {
        'C': [0.01, 0.1, 1, 10],
        'kernel': ['linear', 'poly'],
        'degree': [2, 3, 4]
    }
    scoring = {
        'AUC': 'roc_auc',
        'F1': make_scorer(f1_score),
        'F2': make_scorer(fbeta_score, beta=2)
    }
    grid_search = GridSearchCV(estimator=SVC(probability=True, cache_size=700, max_iter=100000), param_grid=params, scoring=scoring,
                               refit='AUC', return_train_score=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    opt_params = grid_search.best_params_
    grid_search_results = grid_search.cv_results_
    opt_idx = np.argmax(grid_search_results['mean_test_AUC'])
    # print(f'best_params: {opt_params}')

    train_auc = grid_search_results['mean_train_AUC'][opt_idx]
    train_f1 = grid_search_results['mean_train_F1'][opt_idx]
    train_f2 = grid_search_results['mean_train_F2'][opt_idx]
    val_auc = grid_search_results['mean_test_AUC'][opt_idx]
    val_f1 = grid_search_results['mean_test_F1'][opt_idx]
    val_f2 = grid_search_results['mean_test_F2'][opt_idx]

    model = grid_search.best_estimator_
    pred_y = model.predict(test_x)
    # print(pred_y)
    test_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    test_f1 = f1_score(test_y, pred_y)
    test_f2 = fbeta_score(test_y, pred_y, beta=2)

    result = {
        'train-auc': train_auc,
        'train-f1': train_f1,
        'train-f2': train_f2,
        'val-auc': val_auc,
        'val-f1': val_f1,
        'val-f2': val_f2,
        'test-auc': test_auc,
        'test-f1': test_f1,
        'test-f2': test_f2,
        'params': params
    }
    return result


# 3e
def build_nn(train_x, test_x, train_y, test_y):
    params = {
        'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (64, 32), (128, 64), (64, 32, 16), (128, 64, 32)],
        'activation': ['logistic', 'tanh', 'relu'],
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]
    }
    scoring = {
        'AUC': 'roc_auc',
        'F1': make_scorer(f1_score),
        'F2': make_scorer(fbeta_score, beta=2)
    }
    grid_search = GridSearchCV(estimator=MLPClassifier(max_iter=20000), param_grid=params, scoring=scoring,
                               refit='AUC', return_train_score=True, n_jobs=-1)
    grid_search.fit(train_x, train_y)
    opt_params = grid_search.best_params_
    grid_search_results = grid_search.cv_results_
    opt_idx = np.argmax(grid_search_results['mean_test_AUC'])
    # print(f'best_params: {opt_params}')

    train_auc = grid_search_results['mean_train_AUC'][opt_idx]
    train_f1 = grid_search_results['mean_train_F1'][opt_idx]
    train_f2 = grid_search_results['mean_train_F2'][opt_idx]
    val_auc = grid_search_results['mean_test_AUC'][opt_idx]
    val_f1 = grid_search_results['mean_test_F1'][opt_idx]
    val_f2 = grid_search_results['mean_test_F2'][opt_idx]

    model = grid_search.best_estimator_
    pred_y = model.predict(test_x)

    test_auc = roc_auc_score(test_y, model.predict_proba(test_x)[:, 1])
    test_f1 = f1_score(test_y, pred_y)
    test_f2 = fbeta_score(test_y, pred_y, beta=2)

    result = {
        'train-auc': train_auc,
        'train-f1': train_f1,
        'train-f2': train_f2,
        'val-auc': val_auc,
        'val-f1': val_f1,
        'val-f2': val_f2,
        'test-auc': test_auc,
        'test-f1': test_f1,
        'test-f2': test_f2,
        'params': params
    }
    return result
