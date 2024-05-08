import numpy as np
from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def tune_sgtb(x, y, lst_nIter, lst_nu, lst_q, md):
	param_grid = {
		'nIter': lst_nIter,
		'nu': lst_nu,
		'q': lst_q
	}
	grid_search = GridSearchCV(estimator=SGTB(md=md), param_grid=param_grid, scoring='neg_root_mean_squared_error',
							   n_jobs=-1, verbose=2)
	grid_search.fit(x, y)
	opt_params = grid_search.best_params_
	result = {
		'best-nIter': opt_params['nIter'],
		'best-nu': opt_params['nu'],
		'best-q': opt_params['q'],
		'results': grid_search.cv_results_
	}

	return result


def compute_residual(y_true, y_pred):
	return y_true - y_pred


class SGTB(RegressorMixin):
	def __init__(self, nIter=1, q=1, nu=0.1, md=3):
		self.nIter = nIter
		self.q = q
		self.nu = nu
		self.md = md
		self.train_dict = {}
		self.mean = 0
		self.trees = []

	def get_params(self, deep=True):
		return {"nIter": self.nIter,
				"q": self.q,
				"nu": self.nu,
				"md": self.md}

	# set the parameters
	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	# train data using gradient boosting
	def fit(self, x, y):
		self.mean = np.mean(y)
		y_pred = np.full(len(y), self.mean)
		rmse = mean_squared_error(y, y_pred, squared=False)
		self.train_dict[0] = rmse

		residual = compute_residual(y, y_pred)

		for i in range(1, self.nIter + 1):
			# subsample
			idx = np.random.choice(range(len(y)), size=int(self.q * len(y)), replace=False)
			# print(idx.shape)
			# print(x.shape)
			train_x = x[idx, :]
			train_y = y[idx]
			train_residual = residual[idx]

			# train dt on residual
			tree = DecisionTreeRegressor(max_depth=self.md)
			tree.fit(train_x, train_residual)
			self.trees.append(tree)

			# update
			y_pred += self.nu * tree.predict(x)
			residual = compute_residual(y, y_pred)
			rmse = mean_squared_error(y, y_pred, squared=False)

			self.train_dict[i] = rmse

		return self

	def predict(self, x):
		y_pred = np.full(x.shape[0], self.mean)

		for i in range(1, self.nIter + 1):
			pred = self.trees[i-1].predict(x)
			y_pred += self.nu * pred

		return y_pred



