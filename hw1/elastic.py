import numpy as np


def loss(x, y, beta, el, alpha):
    residual = y - np.dot(x, beta)
    residual_l2 = np.linalg.norm(residual)
    beta_l2 = np.linalg.norm(beta)
    beta_l1 = np.linalg.norm(beta, ord=1)

    loss_result = 0.5 * (residual_l2 ** 2) + el * (alpha * (beta_l2 ** 2) + (1 - alpha) * beta_l1)
    return loss_result


def grad_step(x, y, beta, el, alpha, eta):
    gradient = np.zeros_like(beta)
    residual = y - np.dot(x, beta)
    residual_gradient = -x.T.dot(residual)
    l2_gradient = 2 * el * alpha * beta

    for i in range(len(beta)):
        if beta[i] > el * (1 - alpha) * eta:
            prox = beta[i] - el * (1 - alpha) * eta
        elif beta[i] < -1 * el * (1 - alpha) * eta:
            prox = beta[i] + el * (1 - alpha) * eta
        else:
            prox = 0

        gradient[i] = -1 * eta * residual_gradient[i] + prox + -1 * eta * l2_gradient[i]

    return gradient


class ElasticNet:
    def __init__(self, el, alpha, eta, batch, epoch):
        self.el = el
        self.alpha = alpha
        self.eta = eta
        self.batch = batch
        self.epoch = epoch
        self.beta = None

    def coef(self):
        return self.beta

    def train(self, x, y):
        n_sample, n_feature = x.shape
        self.beta = np.zeros(n_feature)
        # self.beta = np.random.rand(n_feature)
        epoch_loss = {}

        for epoch in range(1, self.epoch + 1):
            total_loss = 0.0

            idx = np.random.permutation(n_sample)

            for batch_start in range(0, n_sample, self.batch):
                batch_idx = idx[batch_start: batch_start + self.batch]

                # gradient = grad_step(x[batch_idx], y[batch_idx], self.beta, self.el, self.alpha, self.eta)
                #
                # self.beta -= self.eta * gradient

                self.beta = grad_step(x[batch_idx], y[batch_idx], self.beta, self.el, self.alpha, self.eta)

                total_loss += loss(x[batch_idx], y[batch_idx], self.beta, self.el, self.alpha)

            epoch_loss[epoch] = total_loss / n_sample

        return epoch_loss

    def predict(self, x):
        y_pred = np.dot(x, self.beta)
        return y_pred
