import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


def tune_nn(x, y, hiddenparams, actparams, alphaparams):
    param_grid = {
        'hidden_layer_sizes': hiddenparams,
        'activation': actparams,
        'alpha': alphaparams
    }

    grid_search = GridSearchCV(estimator=MLPClassifier(), param_grid=param_grid, scoring='roc_auc', n_jobs=-1, verbose=2)
    grid_search.fit(x, y)
    opt_params = grid_search.best_params_
    result = {
        'best-hidden': opt_params['hidden_layer_sizes'],
        'best-activation': opt_params['activation'],
        'best-alpha': opt_params['alpha'],
        'results': grid_search.cv_results_
    }

    return result
