import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV


# 2a
def tune_dt(x, y, dparams, lsparams):
    param_grid = {
        'max_depth': dparams,
        'min_samples_leaf': lsparams
    }
    grid_search = GridSearchCV(estimator=tree.DecisionTreeClassifier(), param_grid=param_grid, scoring='roc_auc', return_train_score=True)
    grid_search.fit(x, y)
    opt_params = grid_search.best_params_
    result = {'best-depth': opt_params['max_depth'],
              'best-leaf-samples': opt_params['min_samples_leaf'],
              'results': grid_search.cv_results_}
    return result
