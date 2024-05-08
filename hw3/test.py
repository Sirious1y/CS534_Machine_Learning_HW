import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import roc_auc_score, f1_score, fbeta_score
import pydotplus
import csv
import preprocess
import q2
import perceptron


# 1b partition the data into 70:15:15
def partition_1b(x, y, test_size):
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=test_size)
    return train_x, train_y, test_x, test_y


# encode all categorical data as int
def preprocessing(df):
    x = df.drop('class', axis=1)
    y = df['class']
    # print(x.dtypes)
    # print(y)

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

    return x, y


# 1f perform feature selection
def feature_selection_1f(x, y):
    result = {}
    # compute_correlation
    for method in ['pearson', 'spearman', 'kendall']:
        corr_matrix = preprocess.compute_correlation(x, corrtype=method)
        corr_df = pd.DataFrame(corr_matrix)
        # print(corr_df)
        upper = corr_df.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
        drop = [column for column in upper.columns if any(upper[column] > 0.8)]
        # print(x)
        print(method)
        print(f'columns to drop: {drop}')
        new_x = pd.DataFrame(x).drop(columns=drop, axis=1).to_numpy()
        # print(new_x.shape)
        result.update({method: {'drop': drop, 'new_x': new_x}})

    # rank_correlation
    idx = preprocess.rank_correlation(x, y)
    print('rank_correlation')
    print(f'ranking: {idx}')
    new_x = pd.DataFrame(x)[:][idx[:10]].to_numpy()
    result.update({'rank_corr': {'idx': idx, 'new_x': new_x}})

    # rank_mutual
    idx = preprocess.rank_mutual(x, y)
    print('rank_mutual')
    print(f'ranking: {idx}')
    new_x = pd.DataFrame(x)[:][idx[:10]].to_numpy()
    result.update({'rank_mutual': {'idx': idx, 'new_x': new_x}})

    return result


# 2b grid search for best parameters and plot AUC
def grid_search_2b(train_x, train_y):
    dparams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    lsparams = [1, 5, 10, 50, 100, 150, 200, 300, 500]
    dt_result = q2.tune_dt(train_x, train_y, dparams, lsparams)
    print(f'best-depth: {dt_result["best-depth"]}')
    print(f'best-leaf-samples: {dt_result["best-leaf-samples"]}')
    results = dt_result['results']
    max_depth = []
    min_sample = []
    auc = []
    # print(results.keys())
    for mean_score, params in zip(results['mean_test_score'], results['params']):
        # print(f"Parameters: {params}; Test score: {mean_score} ")
        if params['max_depth'] == dt_result['best-depth'] and params['min_samples_leaf'] == dt_result['best-leaf-samples']:
            opt_auc = mean_score
            continue
        max_depth.append(params['max_depth'])
        min_sample.append(params['min_samples_leaf'])
        auc.append(mean_score)

    # plot 3d scatter plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(max_depth, min_sample, auc)
    # print(opt_auc)
    ax.scatter(dt_result['best-depth'], dt_result['best-leaf-samples'], opt_auc, marker='^')
    ax.set_xlabel('Max_Depth')
    ax.set_ylabel('Min_Sample_Leaf')
    ax.set_zlabel('AUC')
    ax.set_title('Validation AUC Scores on Different Parameters')

    plt.show()


# 2c Re-train
def retrain_2c(train_x, train_y, test_x, test_y, max_depth, min_sample):
    model = tree.DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_sample)
    model.fit(train_x, train_y)
    predict_y = model.predict(test_x)

    auc = roc_auc_score(y_true=test_y, y_score=predict_y)
    f1 = f1_score(y_true=test_y, y_pred=predict_y)
    f2 = fbeta_score(y_true=test_y, y_pred=predict_y, beta=2)

    result = {
        'AUC': auc,
        'F1': f1,
        'F2': f2,
        'model': model
    }
    return result


# 2e Feature selection analyze
def analyze_filtering_2e(train_x, train_y, test_x, test_y, feature_selection):
    dparams = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    lsparams = [1, 5, 10, 50, 100, 150, 200, 300, 500]

    # 1c compute_correlation
    train_x_1c = feature_selection['spearman']['new_x']
    drop = feature_selection['spearman']['drop']
    # test_x_1c = pd.DataFrame(test_x).drop(columns=drop, axis=1).to_numpy()
    parameters_1c = q2.tune_dt(train_x_1c, train_y, dparams, lsparams)
    result_1c = {'method': 'corr_matrix'}
    result_1c.update(retrain_2c(train_x, train_y, test_x, test_y,
                                max_depth=parameters_1c['best-depth'], min_sample=parameters_1c['best-leaf-samples']))
    # del result_1c['model']
    print(result_1c)

    # 1d rank_correlation
    train_x_1d = feature_selection['rank_corr']['new_x']
    idx = feature_selection['rank_corr']['idx']
    # test_x_1d = pd.DataFrame(test_x)[:][idx[:10]].to_numpy()
    parameters_1d = q2.tune_dt(train_x_1d, train_y, dparams, lsparams)
    result_1d = {'method': 'rank_correlation'}
    result_1d.update(retrain_2c(train_x, train_y, test_x, test_y,
                                max_depth=parameters_1d['best-depth'], min_sample=parameters_1d['best-leaf-samples']))
    # del result_1d['model']
    print(result_1d)


    # 1e rank_mutual
    train_x_1e = feature_selection['rank_mutual']['new_x']
    idx = feature_selection['rank_mutual']['idx']
    # test_x_1e = pd.DataFrame(test_x)[:][idx[:10]].to_numpy()
    parameters_1e = q2.tune_dt(train_x_1e, train_y, dparams, lsparams)
    result_1e = {'method': 'rank_mutual'}
    result_1e.update(retrain_2c(train_x, train_y, test_x, test_y,
                                max_depth=parameters_1e['best-depth'], min_sample=parameters_1e['best-leaf-samples']))
    # del result_1e['model']
    print(result_1e)

    columns = ['method', 'AUC', 'F1', 'F2', 'model']
    with open('q2e_result.csv', 'w') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows([result_1c, result_1d, result_1e])



def main():
    # # Q1
    # # Read in data
    # df = pd.read_csv('loan_default.csv')
    # x, y = preprocessing(df)
    # x = x.to_numpy()
    # y = y.to_numpy()
    # print(x.shape)
    # # 1b
    # train_x, train_y, test_x, test_y = partition_1b(x, y, 0.2)
    #
    # # 1f
    # print('1f')
    # feature_selection = feature_selection_1f(train_x, train_y)
    # print('-------------------------------------------------------------------------------------------')
    #
    # # Q2
    #
    # # 2b
    # print('2b')
    # grid_search_2b(train_x, train_y)
    # print('-------------------------------------------------------------------------------------------')
    #
    # # 2c
    # print('2c')
    # result = retrain_2c(train_x, train_y, test_x, test_y, max_depth=4, min_sample=100)
    # print(f'AUC: {result["AUC"]}')
    # print(f'F1: {result["F1"]}')
    # print(f'F2: {result["F2"]}')
    # print('-------------------------------------------------------------------------------------------')
    #
    # # 2d
    # model = result['model']
    # data = tree.export_graphviz(decision_tree=model, max_depth=2)
    # graph = pydotplus.graph_from_dot_data(data)
    # graph.write_png('dt.png')
    #
    # # 2e
    # print('2e')
    # analyze_filtering_2e(train_x, train_y, test_x, test_y, feature_selection)
    # print('-------------------------------------------------------------------------------------------')

    # Q3
    emails, label = perceptron.read_file('spamAssassin.data')
    label = np.array(label)
    # change label to +1 and -1
    label[label == 0] = -1
    # partition
    train_email, train_y, test_email, test_y = partition_1b(emails, label, 0.3)
    # build vocab
    train_x, test_x, vocabulary = perceptron.build_vocab(train_email, test_email, 5)
    print(f'vocabulary size: {len(vocabulary)}')
    # add bias term
    train_x = np.insert(train_x, 0, 1, axis=1)
    test_x = np.insert(test_x, 0, 1, axis=1)

    val_x, val_y, test_x, test_y = partition_1b(test_x, test_y, test_size=0.5)

    # 3g
    max_epochs = [1, 2, 3, 4, 5, 10, 15, 20]
    train_error = []
    val_error = []
    sum_error = []
    for epoch in max_epochs:
        model = perceptron.Perceptron(epoch=epoch)
        train_stats = model.train(train_x, train_y)
        pred_train_y = model.predict(train_x)
        train_error.append(np.sum(np.array(pred_train_y != train_y))/train_x.shape[0])
        pred_val_y = model.predict(val_x)
        val_error.append(np.sum(np.array(pred_val_y != val_y))/val_x.shape[0])
        if train_stats[max(train_stats.keys())] == 0:
            total_mistake = sum(train_stats.values())
            print(total_mistake)

    print('3g')
    print('Perceptron performances')
    print(train_error)
    print(val_error)

    plt.plot(max_epochs, train_error, label='train error')
    plt.plot(max_epochs, val_error, label='estimated error')
    plt.xlabel('Max_Epochs')
    plt.ylabel('Error Rate (# of error / sample size)')
    plt.title('Training and estimated error rate with respect to max epochs')
    plt.legend()
    plt.show()
    print('-------------------------------------------------------------------------------------------')

    # 3k
    max_epochs = [1, 2, 3, 4, 5, 10, 15, 20, 25, 30]
    train_error = []
    val_error = []
    sum_error = []
    for epoch in max_epochs:
        model = perceptron.AvgPerceptron(epoch=epoch)
        train_stats = model.train(train_x, train_y)
        pred_train_y = model.predict(train_x)
        train_error.append(np.sum(np.array(pred_train_y != train_y)) / train_x.shape[0])
        pred_val_y = model.predict(val_x)
        val_error.append(np.sum(np.array(pred_val_y != val_y)) / val_x.shape[0])
        if train_stats[max(train_stats.keys())] == 0:
            total_mistake = sum(train_stats.values())
            print(total_mistake)

    print('3k')
    print('Averaged perceptron performances')
    print(train_error)
    print(val_error)

    plt.plot(max_epochs, train_error, label='train error')
    plt.plot(max_epochs, val_error, label='estimated error')
    plt.xlabel('Max_Epochs')
    plt.ylabel('Error Rate (# of error / sample size)')
    plt.title('Training and estimated error rate w.r.t. max epochs (average perceptron)')
    plt.legend()
    plt.show()
    print('-------------------------------------------------------------------------------------------')

    # 3l
    print('3l')
    perc = perceptron.Perceptron(5)
    train_x_3l = np.concatenate((train_x, val_x))
    train_y_3l = np.concatenate((train_y, val_y))
    perc.train(train_x_3l, train_y_3l)
    pred_y = perc.predict(test_x)
    test_error = np.sum(np.array(pred_y != test_y)) / test_x.shape[0]
    print(f'Expected error rate: {test_error}')
    print('-------------------------------------------------------------------------------------------')

    # 3m
    print('3m')
    weights = perc.get_weight()
    word_weight = {}
    for word in vocabulary.keys():
        w = weights[vocabulary[word]]
        word_weight[word] = w

    descending = sorted(word_weight, key=word_weight.get, reverse=True)[:15]
    ascending = sorted(word_weight, key=word_weight.get)[:15]

    print(f'positive top 15: {descending}')
    print(f'negative top 15: {ascending}')

    return


if __name__ == '__main__':
    main()
