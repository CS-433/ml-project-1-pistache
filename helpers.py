"""Some helper functions for project 1."""
import csv
import numpy as np
from implementations import mean_squared_error_gd, mean_squared_error_sgd, least_squares, ridge_regression, \
    logistic_regression, reg_logistic_regression, predict_simple, predict_logistic, penalized_logistic_regression


def load_csv_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), tX (features) and ids (event ids)"""
    y = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=str, usecols=1)
    x = np.genfromtxt(data_path, delimiter=",", skip_header=1)
    ids = x[:, 0].astype(np.int)
    input_data = x[:, 2:]

    # convert class labels from strings to binary (-1,1)
    yb = np.ones(len(y))
    yb[np.where(y == "b")] = -1

    # sub-sample
    if sub_sample:
        yb = yb[::50]
        input_data = input_data[::50]
        ids = ids[::50]

    return yb, input_data, ids


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments: ids (event ids associated with each prediction)
               y_pred (predicted class labels)
               name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def fill_missing(x, col_idx, func):
    replace_val = func(x[x[:, col_idx] != -999, col_idx])
    x[x[:, col_idx] == -999, col_idx] = replace_val
    return x


def partition_data(x, y, ids):
    xs, ys, idss = [], [], []
    population = []
    for i in range(4):
        xs.append(x[x[:, 22] == i])
        ys.append(y[x[:, 22] == i])
        idss.append(ids[x[:, 22] == i])
        # population.append(xs[-1].shape[0] / x.shape[0])
    # population = np.array(population)
    return xs, ys, idss


def add_bias(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], 1)


def sigmoid(t):
    return 1 / (1 + np.exp(-t))


def process_features(x, max_degree=6):
    # fill first column
    x = fill_missing(x, col_idx=0, func=np.median)

    # remove other meaningless columns
    var = np.var(x, axis=0)
    removing_cols = np.where(var == 0)[0]
    x = np.delete(x, removing_cols, 1)

    x = remove_outliers(x)

    # normalize
    x -= np.mean(x, axis=0)
    x /= np.std(x, axis=0)

    # add poly
    x_copy = x.copy()
    for k in range(x_copy.shape[1]):
        for j in range(k, x_copy.shape[1]):
            new_col = x_copy[:, k] * x_copy[:, j]
            new_col = np.reshape(new_col, (x.shape[0], -1))
            x = np.concatenate([x, new_col], 1)
    x = np.concatenate([x, np.sin(x_copy), np.cos(x_copy)], 1)
    x = np.concatenate([x, x_copy ** 3, x_copy ** 4, x_copy ** 5, x_copy ** 6], 1)
    return x


def remove_outliers(x):
    out = x.copy()
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    lower = q25 - 1.5 * iqr
    upper = q75 + 1.5 * iqr

    for i in range(x.shape[1]):
        col = out[:, i]
        out[:, i][(col > upper) | (col < lower)] = np.median(col)
    return out


def build_k_indices(N, k_fold):
    num_row = N
    interval = int(num_row / k_fold)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def do_cross_validation(x, y, nfolds=4):
    lambdas = np.logspace(-4, 0, 10)

    accs = []
    losses = []
    for lambda_ in lambdas:
        N = x.shape[0]

        k_indices = build_k_indices(N, nfolds)

        validation_accuracy_k = []
        validation_loss_k = []

        for k in range(nfolds):
            x_validation_k = x[k_indices[k]]
            y_validation_k = y[k_indices[k]]
            x_train_k = np.delete(x, k_indices[k], axis=0)
            y_train_k = np.delete(y, k_indices[k], axis=0)

            x_train_k, x_validation_k = add_bias(x_train_k), add_bias(x_validation_k)

            w, loss = ridge_regression(y_train_k, x_train_k, lambda_)

            validation_accuracy_k.append((predict_simple(x_validation_k, w) == y_validation_k).mean())
            validation_loss_k.append(loss)

        accs.append(np.mean(validation_accuracy_k))
        losses.append(np.mean(validation_loss_k))

    idx = np.argmin(losses)
    best_loss = losses[idx]
    best_lambda = lambdas[idx]

    return best_lambda, best_loss
