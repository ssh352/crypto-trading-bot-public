import itertools
import math

import numpy as np
import pandas as pd
import psycopg2
import torch as T
from sklearn.model_selection import train_test_split

device = T.device("cpu")  # apply to Tensor or Module


class PairTradesDataset(T.utils.data.Dataset):

    def __init__(self, x_data, y_data, num_rows=None):
        self.x_data = T.tensor(x_data,
                               dtype=T.float32).to(device)
        self.y_data = T.tensor(y_data,
                               dtype=T.float32).to(device)
        self.y_data = self.y_data.reshape(-1, 1)

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        if T.is_tensor(idx):
            idx = idx.tolist()
        preds = self.x_data[idx, :]  # idx rows, all 4 cols
        lbl = self.y_data[idx, :]    # idx rows, the 1 col

        sample = {'predictors': preds, 'target': lbl}

        return sample


class Net(T.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hid1 = T.nn.Linear(124, 84)
        self.hid2 = T.nn.Linear(84, 58)
        self.hid3 = T.nn.Linear(58, 2)
        self.oupt = T.nn.Linear(2, 1)

        T.nn.init.xavier_uniform_(self.hid1.weight)
        T.nn.init.zeros_(self.hid1.bias)
        T.nn.init.xavier_uniform_(self.hid2.weight)
        T.nn.init.zeros_(self.hid2.bias)
        T.nn.init.xavier_uniform_(self.hid3.weight)
        T.nn.init.zeros_(self.hid3.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        z = T.sigmoid(self.hid1(x))
        z = T.sigmoid(self.hid2(z))
        z = T.sigmoid(self.hid3(z))
        z = T.sigmoid(self.oupt(z))
        return z


def accuracy(model, ds):
    # ds is a iterable Dataset of Tensors
    n_correct = 0
    n_wrong = 0

    # alt: create DataLoader and then enumerate it
    for i in range(len(ds)):
        inpts = ds[i]['predictors']
        target = ds[i]['target']    # float32  [0.0] or [1.0]
        with T.no_grad():
            oupt = model(inpts)

        # avoid 'target == 1.0'
        if target < 0.5 and oupt < 0.5:  # .item() not needed
            n_correct += 1
        elif target >= 0.5 and oupt >= 0.5:
            n_correct += 1
        else:
            n_wrong += 1

    return (n_correct * 1.0) / (n_correct + n_wrong)


def main():
    conn = psycopg2.connect(
        host='',
        database='',
        user='',
        password='')

    cursor = conn.cursor()
    cursor.itersize = 10
    cursor.execute(
        'SELECT features, submit_timestamp, bitmex_fill_timestamp, deribit_fill_timestamp FROM pair_trades_ml WHERE quantity > 0 LIMIT 1000')

    X_data = []
    y_data = []

    for row in cursor:
        [bitmex_features, deribit_features] = row[0]
        bitmex_features = list(map(lambda x: x['value'], bitmex_features))
        deribit_features = list(map(lambda x: x['value'], deribit_features))
        submit_ts = row[1]
        bitmex_fill_ts = row[2]
        deribit_fill_ts = row[3]

        is_filled = False

        if bitmex_fill_ts is not None and deribit_fill_ts is not None:
            both_filled_ts = max(bitmex_fill_ts, deribit_fill_ts)
            time_to_fill_s = (both_filled_ts - submit_ts).total_seconds()

            if time_to_fill_s <= 10:
                is_filled = True

        contains_null_feature = False

        for feature in itertools.chain(bitmex_features, deribit_features):
            if feature is None:
                contains_null_feature = True
                break

        if contains_null_feature:
            continue

        X_data.append(bitmex_features + deribit_features)
        y_data.append(1 if is_filled else 0)

    # Begin neural network
    T.manual_seed(1)
    np.random.seed(1)

    bat_size = 10
    net = Net().to(device)

    # Standardization
    X_tensor = T.tensor(X_data)
    X_means = X_tensor.mean(dim=0)
    X_stds = X_tensor.std(dim=0, unbiased=False)  # ? Should be unbiased?
    X_tensor_std = (X_tensor - X_means) / X_stds
    X_tensor_min = X_tensor_std.min(dim=0)
    X_tensor_max = X_tensor_std.max(dim=0)
    X_tensor_range = X_tensor_max.values - X_tensor_min.values
    X_tensor_std_norm = (X_tensor_std - X_tensor_min.values) / X_tensor_range
    X_data_std_norm = X_tensor_std_norm.tolist()

    print(X_tensor_std_norm.shape)


if __name__ == '__main__':
    main()
