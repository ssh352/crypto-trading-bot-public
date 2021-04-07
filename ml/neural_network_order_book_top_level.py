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
        self.hid_1_1 = T.nn.Linear(2, 2)
        self.hid_1_2 = T.nn.Linear(2, 2)
        self.hid_2_1 = T.nn.Linear(2, 1)
        self.hid_2_2 = T.nn.Linear(2, 1)
        self.oupt = T.nn.Linear(2, 1)

        T.nn.init.xavier_uniform_(self.hid_1_1.weight)
        T.nn.init.zeros_(self.hid_1_1.bias)
        T.nn.init.xavier_uniform_(self.hid_1_2.weight)
        T.nn.init.zeros_(self.hid_1_2.bias)
        T.nn.init.xavier_uniform_(self.hid_2_1.weight)
        T.nn.init.zeros_(self.hid_2_1.bias)
        T.nn.init.xavier_uniform_(self.hid_2_2.weight)
        T.nn.init.zeros_(self.hid_2_2.bias)
        T.nn.init.xavier_uniform_(self.oupt.weight)
        T.nn.init.zeros_(self.oupt.bias)

    def forward(self, x):
        dim = 1
        if len(x.shape) == 1:
            dim = 0
            size = x.shape[0] / 2
        else:
            size = x.shape[1] / 2
        x1, x2 = T.split(x, int(size), dim=dim)
        z1 = T.relu(self.hid_1_1(x1))
        z2 = T.relu(self.hid_1_2(x2))
        z1 = T.relu(self.hid_2_1(z1))
        z2 = T.relu(self.hid_2_2(z2))
        z = T.cat([z1, z2], dim=dim)
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
        'SELECT features, submit_timestamp, bitmex_fill_timestamp, deribit_fill_timestamp FROM pair_trades_ml WHERE quantity > 0')

    X_data = []
    y_data = []

    # chosen_features = ['bid_ask_spread', 'top_5_bid_ask_imbalance', 'volume_order_imbalance_1000_ms_5_level',
    #                    'trade_flow_imbalance_60s', 'midprice_variance_60', 'signed_trade_size_variance_60']
    chosen_features = ['midprice_variance_60', 'signed_trade_size_variance_60']

    for row in cursor:
        X1_datum = []
        X2_datum = []

        # [bitmex_order_book, deribit_order_book] = row[0]
        [bitmex_features, deribit_features] = row[0]

        contains_null = False

        for bitmex_feature, deribit_feature in zip(bitmex_features, deribit_features):
            if bitmex_feature['name'] == deribit_feature['name'] and bitmex_feature['name'] in chosen_features:
                contains_null = bitmex_feature['value'] == None or deribit_feature['value'] == None
                if contains_null:
                    break

                X1_datum.append(bitmex_feature['value'])
                X2_datum.append(deribit_feature['value'])

        if contains_null:
            continue

        # bitmex_bid_price = bitmex_order_book[0][0][0]['price_x100'] / 100
        # bitmex_bid_volume = bitmex_order_book[0][0][1]
        # bitmex_ask_price = bitmex_order_book[1][0][0]['price_x100'] / 100
        # bitmex_ask_volume = bitmex_order_book[1][0][1]

        # deribit_bid_price = deribit_order_book[0][0][0]['price_x100'] / 100
        # deribit_bid_volume = deribit_order_book[0][0][1]
        # deribit_ask_price = deribit_order_book[1][0][0]['price_x100'] / 100
        # deribit_ask_volume = deribit_order_book[1][0][1]

        submit_ts = row[1]
        bitmex_fill_ts = row[2]
        deribit_fill_ts = row[3]

        is_filled = False

        if bitmex_fill_ts is not None and deribit_fill_ts is not None:
            both_filled_ts = max(bitmex_fill_ts, deribit_fill_ts)
            time_to_fill_s = (both_filled_ts - submit_ts).total_seconds()

            if time_to_fill_s <= 10:
                is_filled = True

        # X_datum.append(bitmex_bid_price)
        # X_datum.append(bitmex_bid_volume)
        # X_datum.append(bitmex_ask_price)
        # X_datum.append(bitmex_ask_volume)
        # X_datum.append(deribit_bid_price)
        # X_datum.append(deribit_bid_volume)
        # X_datum.append(deribit_ask_price)
        # X_datum.append(deribit_ask_volume)

        X_data.append(X1_datum + X2_datum)

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
    X_data_std_norm = X_tensor_std_norm.tolist()  # TODO: Feature scaling

    X_train, X_test, y_train, y_test = train_test_split(
        X_data_std_norm, y_data, test_size=0.2)

    train_ds = PairTradesDataset(
        X_train, y_train)  # all rows
    test_ds = PairTradesDataset(X_test, y_test)

    bat_size = 10
    train_ldr = T.utils.data.DataLoader(train_ds,
                                        batch_size=bat_size, shuffle=True)

    # Train neural network
    print("\nPreparing training")
    net = net.train()  # set training mode
    lrn_rate = 0.01
    loss_obj = T.nn.BCELoss()  # binary cross entropy
    optimizer = T.optim.SGD(net.parameters(),
                            lr=lrn_rate)
    max_epochs = 1000
    ep_log_interval = 10
    print("Loss function: " + str(loss_obj))
    print("Optimizer: SGD")
    print("Learn rate: 0.01")
    print("Batch size: 10")
    print("Max epochs: " + str(max_epochs))

    print("\nStarting training")
    for epoch in range(0, max_epochs):
        epoch_loss = 0.0            # for one full epoch
        epoch_loss_custom = 0.0
        num_lines_read = 0

        for (batch_idx, batch) in enumerate(train_ldr):
            X = batch['predictors']  # [10,4]  inputs
            Y = batch['target']      # [10,1]  targets
            oupt = net(X)            # [10,1]  computeds

            loss_val = loss_obj(oupt, Y)   # a tensor
            epoch_loss += loss_val.item()  # accumulate
            # epoch_loss += loss_val  # is OK
            # epoch_loss_custom += my_bce(net, batch)

            optimizer.zero_grad()  # reset all gradients
            loss_val.backward()   # compute all gradients
            optimizer.step()      # update all weights

        if epoch % ep_log_interval == 0:
            print("epoch = %4d   loss = %0.4f" %
                  (epoch, epoch_loss))
            # print("custom loss = %0.4f" % epoch_loss_custom)
            # print("")
    print("Done ")

    # 4. evaluate model
    net = net.eval()
    acc_train = accuracy(net, train_ds)
    print('\ny_train non-fills/total = %0.2f%%' %
          ((len(y_train) - sum(y_train)) / len(y_train) * 100))
    print('y_test non-fills/total = %0.2f%%' %
          ((len(y_test) - sum(y_test)) / len(y_test) * 100))
    print("\nAccuracy on train data = %0.2f%%" %
          (acc_train * 100))
    acc_test = accuracy(net, test_ds)
    print("Accuracy on test data = %0.2f%%" %
          (acc_test * 100))


if __name__ == "__main__":
    main()
