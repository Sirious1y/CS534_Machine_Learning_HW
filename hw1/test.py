import numpy as np
import pandas as pd
import q2
import elastic
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def main():
    # read the data from csv files
    train_x = pd.read_csv('energydata\energy_train.csv')
    val_x = pd.read_csv('energydata\energy_val.csv')
    test_x = pd.read_csv('energydata\energy_test.csv')

    # create true values y
    train_y = train_x['Appliances']
    val_y = val_x['Appliances']
    test_y = test_x['Appliances']

    # drop irrelevant features
    train_x = train_x.drop(['date', 'Appliances'], axis=1)
    val_x = val_x.drop(['date', 'Appliances'], axis=1)
    test_x = test_x.drop(['date', 'Appliances'], axis=1)

    feature_names = list(train_x.columns)
    # print(feature_names)

    # 2b) preprocess
    train_x = train_x.to_numpy()
    val_x = val_x.to_numpy()
    test_x = test_x.to_numpy()
    train_x, val_x, text_x = q2.preprocess_data(train_x, val_x, test_x)
    # print(np.mean(train_x), min(train_x[:, 1]))
    # print(np.mean(train_x, axis=1))

    train_y = train_y.to_numpy()
    val_y = val_y.to_numpy()
    test_y = test_y.to_numpy()

    # 2c) eval_linear1
    lr1_result = q2.eval_linear1(train_x, train_y, val_x, val_y, test_x, test_y)
    print('lr1:', lr1_result)
    # 2d) eval_linear2
    lr2_result = q2.eval_linear2(train_x, train_y, val_x, val_y, test_x, test_y)
    print('lr2:', lr2_result)
    print('-------------------------------------------------------------------------')

    # 2e) Report performances of eval_linear1 and eval_linear2
    columns = ['train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']
    with open('q2_lr_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([lr1_result, lr2_result])

    # 2f) eval_ridge1
    ridge1_result = q2.eval_ridge1(train_x, train_y, val_x, val_y, test_x, test_y, alpha=1)
    print('ridge1:', ridge1_result)
    # 2g) eval_lasso1
    lasso1_result = q2.eval_lasso1(train_x, train_y, val_x, val_y, test_x, test_y, alpha=1)
    print('lasso1:', lasso1_result)
    print('--------------------------------------------------------------------------')

    # 2h) different alpha values
    # did not use eval_ridge1 and eval_lasso1 because I want to access the coefficients for 2k)
    # RIDGE
    alpha_ridge = np.logspace(-5, 6, 12)
    # alpha_ridge = [0.1, 0.5, 1, 5, 10, 20, 30, 40, 50, 100, 1000, 10000, 100000]
    alpha_ridge_result = []
    ridge_coefs = []
    for alpha in alpha_ridge:
        result = {
            'alpha': alpha
        }
        model = linear_model.Ridge(alpha=alpha)

        model.fit(train_x, train_y)
        ridge_coefs.append(model.coef_)

        # test model on train, val, and test sets separately
        train_y_pred = model.predict(train_x)
        val_y_pred = model.predict(val_x)
        test_y_pred = model.predict(test_x)

        # calculate metrics
        train_rmse = mean_squared_error(y_true=train_y, y_pred=train_y_pred, squared=False)
        val_rmse = mean_squared_error(y_true=val_y, y_pred=val_y_pred, squared=False)
        test_rmse = mean_squared_error(y_true=test_y, y_pred=test_y_pred, squared=False)

        train_r2 = r2_score(y_true=train_y, y_pred=train_y_pred)
        val_r2 = r2_score(y_true=val_y, y_pred=val_y_pred)
        test_r2 = r2_score(y_true=test_y, y_pred=test_y_pred)

        result.update({
            'train-rmse': train_rmse,
            'train-r2': train_r2,
            'val-rmse': val_rmse,
            'val-r2': val_r2,
            'test-rmse': test_rmse,
            'test-r2': test_r2
        })
        alpha_ridge_result.append(result)

    print('ridge_alpha:', alpha_ridge_result)
    columns = ['alpha', 'train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']
    with open('q2_ridge_alpha.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(alpha_ridge_result)
    # -------------------------------------------------------------------------------------------
    # LASSO
    alpha_lasso = [0.1, 0.5, 1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 100]
    alpha_lasso_result = []
    lasso_coefs = []
    for alpha in alpha_lasso:
        result = {
            'alpha': alpha
        }
        model = linear_model.Lasso(alpha=alpha)

        model.fit(train_x, train_y)
        # print(f'alpha {alpha} finished')
        lasso_coefs.append(model.coef_)

        # test model on train, val, and test sets separately
        train_y_pred = model.predict(train_x)
        val_y_pred = model.predict(val_x)
        test_y_pred = model.predict(test_x)

        # calculate metrics
        train_rmse = mean_squared_error(y_true=train_y, y_pred=train_y_pred, squared=False)
        val_rmse = mean_squared_error(y_true=val_y, y_pred=val_y_pred, squared=False)
        test_rmse = mean_squared_error(y_true=test_y, y_pred=test_y_pred, squared=False)

        train_r2 = r2_score(y_true=train_y, y_pred=train_y_pred)
        val_r2 = r2_score(y_true=val_y, y_pred=val_y_pred)
        test_r2 = r2_score(y_true=test_y, y_pred=test_y_pred)

        result.update({
            'train-rmse': train_rmse,
            'train-r2': train_r2,
            'val-rmse': val_rmse,
            'val-r2': val_r2,
            'test-rmse': test_rmse,
            'test-r2': test_r2
        })
        alpha_lasso_result.append(result)

    print('lasso_alpha:', alpha_lasso_result)

    columns = ['alpha', 'train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']
    with open('q2_lasso_alpha.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(alpha_lasso_result)

    print('-------------------------------------------------------------------------')
    # based on the data from previous questions
    opt_ridge = 1000
    opt_lasso = 3

    # 2i) eval_ridge2 & eval_lasso2
    ridge2_result = q2.eval_ridge2(train_x, train_y, val_x, val_y, test_x, test_y, alpha=opt_ridge)
    print('ridge2:', ridge2_result)

    lasso2_result = q2.eval_lasso2(train_x, train_y, val_x, val_y, test_x, test_y, alpha=opt_lasso)
    print('lasso2:', lasso2_result)
    print('-------------------------------------------------------------------------')

    # 2j) report performances of eval_ridge2 & eval_lasso2
    columns = ['model', 'train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']
    ridge2 = {
        'model': 'Ridge'
    }
    lasso2 = {
        'model': 'Lasso'
    }
    ridge2.update(ridge2_result)
    lasso2.update(lasso2_result)
    with open('q2_ridgelasso2.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([ridge2, lasso2])

    # ---------------------------------------------------------------------------------

    # 2k) Coefficient path plots
    # RIDGE
    plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    plt.title("Ridge Coefficient Path")
    plt.plot(alpha_ridge, ridge_coefs, label=feature_names)
    plt.axvline(x=opt_ridge, color='r', linestyle='--')
    plt.text(opt_ridge, -80, 'opt alpha', color='r')
    plt.xscale('log')
    plt.xticks(alpha_ridge)
    plt.xlabel('Alpha')
    plt.ylabel('Ridge Coefficients')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper left')
    plt.show()

    plt.clf()
    # LASSO
    plt.figure(figsize=(12, 6))
    # plt.subplot(121)
    plt.title("Lasso Coefficient Path")
    plt.plot(alpha_lasso, lasso_coefs, label=feature_names)
    plt.axvline(x=opt_lasso, color='r', linestyle='--')
    plt.text(opt_lasso, -65, 'opt alpha', color='r')
    plt.xscale('log')
    # plt.xticks(alpha_lasso)
    plt.xlabel('Alpha')
    plt.ylabel('Lasso Coefficients')
    plt.grid(True)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

    plt.clf()
    # -----------------------------------------------------------------------------------
    # 3f)
    # el is opt_ridge and opt_lasso
    alpha = 0.5
    eta_ridge = [0.0001, 0.00005, 0.00001, 0.000005, 0.000001]
    eta_lasso = [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    batch_size_ridge = 300
    batch_size_lasso = 100
    epoch = 10
    # print(train_x.shape)
    elastic_ridge_result = []
    elastic_lasso_result = []
    for i in range(5):
        model = elastic.ElasticNet(el=opt_ridge, alpha=alpha, eta=eta_ridge[i], batch=batch_size_ridge, epoch=epoch)
        temp_ridge_result = model.train(train_x, train_y)
        # print(f'complete {i}th rate for ridge')
        model = elastic.ElasticNet(el=opt_lasso, alpha=alpha, eta=eta_lasso[i], batch=batch_size_lasso, epoch=epoch)
        temp_lasso_result = model.train(train_x, train_y)
        # print(f'complete {i}th rate for lasso')
        ridge_result_list = []
        lasso_result_list = []
        for j in range(1, epoch + 1):
            ridge_result_list.append(temp_ridge_result[j])
            lasso_result_list.append(temp_lasso_result[j])
        # print(f'lr={eta_ridge[i]}and{eta_lasso[i]}:')
        # print(ridge_result_list)
        # print(lasso_result_list)
        elastic_ridge_result.append(np.array(ridge_result_list))
        elastic_lasso_result.append(np.array(lasso_result_list))

    # Ridge graph
    plt.figure(figsize=(12, 6))
    # plt.yscale('log')
    for i in range(len(eta_ridge)):
        plt.plot(range(1, epoch + 1), elastic_ridge_result[i], label=f'lr={eta_ridge[i]}')
    plt.title(f'ElasticNet Training Loss with el={opt_ridge}, alpha=0.5, batch=300')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.tight_layout()
    plt.legend()
    plt.show()

    plt.clf()
    # Lasso graph
    plt.figure(figsize=(12,6))
    for i in range(len(eta_lasso)):
        plt.plot(range(1, epoch + 1), elastic_lasso_result[i], label=f'lr={eta_lasso[i]}')
    plt.title(f'ElasticNet Training Loss with el={opt_lasso}, alpha=0.5, batch=100')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    # plt.yscale('log')
    plt.tight_layout()
    plt.legend()
    plt.show()
    plt.close()

    # from the graphs, we can have the optimal lr
    ridge_lr = 0.000005
    lasso_lr = 0.0001

    elastic_ridge = elastic.ElasticNet(el=opt_ridge, alpha=alpha, eta=ridge_lr, batch=batch_size_ridge, epoch=epoch)
    elastic_lasso = elastic.ElasticNet(el=opt_lasso, alpha=alpha, eta=lasso_lr, batch=batch_size_lasso, epoch=epoch)

    elastic_ridge.train(train_x, train_y)
    elastic_lasso.train(train_x, train_y)

    elastic_ridge_trainpred = elastic_ridge.predict(train_x)
    elastic_ridge_valpred = elastic_ridge.predict(val_x)
    elastic_ridge_testpred = elastic_ridge.predict(test_x)

    elastic_lasso_trainpred = elastic_lasso.predict(train_x)
    elastic_lasso_valpred = elastic_lasso.predict(val_x)
    elastic_lasso_testpred = elastic_lasso.predict(test_x)

    elastic_ridge_temp = {
        'train-rmse': mean_squared_error(y_true=train_y, y_pred=elastic_ridge_trainpred, squared=False),
        'train-r2': r2_score(y_true=train_y, y_pred=elastic_ridge_trainpred),
        'val-rmse': mean_squared_error(y_true=val_y, y_pred=elastic_ridge_valpred, squared=False),
        'val-r2': r2_score(y_true=val_y, y_pred=elastic_ridge_valpred),
        'test-rmse': mean_squared_error(y_true=test_y, y_pred=elastic_ridge_testpred, squared=False),
        'test-r2': r2_score(y_true=test_y, y_pred=elastic_ridge_testpred)
    }

    elastic_lasso_temp = {
        'train-rmse': mean_squared_error(y_true=train_y, y_pred=elastic_lasso_trainpred, squared=False),
        'train-r2': r2_score(y_true=train_y, y_pred=elastic_lasso_trainpred),
        'val-rmse': mean_squared_error(y_true=val_y, y_pred=elastic_lasso_valpred, squared=False),
        'val-r2': r2_score(y_true=val_y, y_pred=elastic_lasso_valpred),
        'test-rmse': mean_squared_error(y_true=test_y, y_pred=elastic_lasso_testpred, squared=False),
        'test-r2': r2_score(y_true=test_y, y_pred=elastic_lasso_testpred)
    }

    columns = ['model', 'train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']
    elastic_ridge_result = {
        'model': 'el=1000, lr=0.000005'
    }
    elastic_lasso_result = {
        'model': 'el=3, lr=0.0001'
    }
    elastic_ridge_result.update(elastic_ridge_temp)
    elastic_lasso_result.update(elastic_lasso_temp)
    print(elastic_ridge_result)
    print(elastic_lasso_result)
    with open('q3_elastic_opt_lr.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([elastic_ridge_result, elastic_lasso_result])

    print('--------------------------------------------------------------------------')

    # 3g)
    alphas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    result_3g_ridge = []
    result_3g_lasso = []
    for alpha in alphas:
        elastic_ridge = elastic.ElasticNet(el=opt_ridge, alpha=alpha, eta=ridge_lr, batch=batch_size_ridge, epoch=epoch)
        elastic_lasso = elastic.ElasticNet(el=opt_lasso, alpha=alpha, eta=lasso_lr, batch=batch_size_lasso, epoch=epoch)
        elastic_ridge.train(train_x, train_y)
        elastic_lasso.train(train_x, train_y)

        elastic_ridge_trainpred = elastic_ridge.predict(train_x)
        elastic_ridge_valpred = elastic_ridge.predict(val_x)
        elastic_ridge_testpred = elastic_ridge.predict(test_x)

        elastic_lasso_trainpred = elastic_lasso.predict(train_x)
        elastic_lasso_valpred = elastic_lasso.predict(val_x)
        elastic_lasso_testpred = elastic_lasso.predict(test_x)

        elastic_ridge_temp = {
            'train-rmse': mean_squared_error(y_true=train_y, y_pred=elastic_ridge_trainpred, squared=False),
            'train-r2': r2_score(y_true=train_y, y_pred=elastic_ridge_trainpred),
            'val-rmse': mean_squared_error(y_true=val_y, y_pred=elastic_ridge_valpred, squared=False),
            'val-r2': r2_score(y_true=val_y, y_pred=elastic_ridge_valpred),
            'test-rmse': mean_squared_error(y_true=test_y, y_pred=elastic_ridge_testpred, squared=False),
            'test-r2': r2_score(y_true=test_y, y_pred=elastic_ridge_testpred)
        }

        elastic_lasso_temp = {
            'train-rmse': mean_squared_error(y_true=train_y, y_pred=elastic_lasso_trainpred, squared=False),
            'train-r2': r2_score(y_true=train_y, y_pred=elastic_lasso_trainpred),
            'val-rmse': mean_squared_error(y_true=val_y, y_pred=elastic_lasso_valpred, squared=False),
            'val-r2': r2_score(y_true=val_y, y_pred=elastic_lasso_valpred),
            'test-rmse': mean_squared_error(y_true=test_y, y_pred=elastic_lasso_testpred, squared=False),
            'test-r2': r2_score(y_true=test_y, y_pred=elastic_lasso_testpred)
        }
        result_ridge = {
            'alpha': alpha
        }
        result_lasso = {
            'alpha': alpha
        }
        result_ridge.update(elastic_ridge_temp)
        result_lasso.update(elastic_lasso_temp)
        result_3g_ridge.append(result_ridge)
        result_3g_lasso.append(result_lasso)

    print(result_3g_ridge)
    print(result_3g_lasso)

    columns = ['alpha', 'train-rmse', 'train-r2', 'val-rmse', 'val-r2', 'test-rmse', 'test-r2']

    with open('q3_elastic_ridge_alphas.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_3g_ridge)

    with open('q3_elastic_lasso_alphas.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(result_3g_lasso)

    print('------------------------------------------------------------------')

    # 3i)
    elastic_best_test = elastic.ElasticNet(el=3, alpha=0.2, eta=0.0001, batch=batch_size_lasso, epoch=100)
    elastic_best_val = elastic.ElasticNet(el=3, alpha=0.4, eta=0.0001, batch=batch_size_lasso, epoch=100)
    elastic_best_test.train(train_x, train_y)
    elastic_best_val.train(train_x, train_y)

    best_test_coef = elastic_best_test.coef()
    best_val_coef = elastic_best_val.coef()

    # print(best_test_coef)
    # print(best_val_coef)
    result_3i = {
        'test_coef': best_test_coef,
        'val_coef': best_val_coef
    }
    result_3i = pd.DataFrame(data=result_3i, index=feature_names)
    print(result_3i)
    result_3i.to_csv('q3_elastic_coef.csv')


if __name__ == "__main__":
    main()
