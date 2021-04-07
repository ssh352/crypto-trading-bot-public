import math
import pickle

import numpy as np
import pandas as pd
import psycopg2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

chosen_features = ['bid_ask_spread',
                   'top_1_quantity_same_side',
                   'top_1_quantity_opposite_side',
                   'top_1_bid_ask_imbalance',
                   'bid_imbalance_10_to_15',
                   'ask_imbalance_10_to_15',
                   'volume_order_imbalance_1000_ms_1_level',
                   'trade_flow_imbalance_60s',
                   'css_1000',
                   'relative_strength_60_1',
                   'ln_return_6000',
                   'midprice_variance_60',
                   'signed_trade_size_variance_60']


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

    col_names = []
    X_data = []
    y_data = []

    for row in cursor:
        X_datum = []

        contains_invalid_value = False

        # [bitmex_order_book, deribit_order_book] = row[0]
        for i, features in enumerate(row[0]):
            exchange_prefix = 'bitmex_' if i == 0 else 'deribit_'

            for feature in features:
                if feature['name'] not in chosen_features:
                    continue

                if exchange_prefix + feature['name'] not in col_names:
                    col_names.append(exchange_prefix + feature['name'])

                if feature['value'] is None or np.isnan(feature['value'] or np.isinf(feature['value'])):
                    contains_invalid_value = True
                    break

                X_datum.append(feature['value'])

            if contains_invalid_value:
                break
        if contains_invalid_value:
            continue

        submit_ts = row[1]
        bitmex_fill_ts = row[2]
        deribit_fill_ts = row[3]

        is_filled = False

        if bitmex_fill_ts is not None and deribit_fill_ts is not None:
            both_filled_ts = max(bitmex_fill_ts, deribit_fill_ts)
            time_to_fill_s = (both_filled_ts - submit_ts).total_seconds()

        if time_to_fill_s <= 10:
            is_filled = True

        X_data.append(X_datum)
        y_data.append(1 if is_filled else 0)

    X_df = pd.DataFrame(X_data, columns=col_names)

    # VIF
    vif = pd.DataFrame()
    vif['features'] = X_df.columns
    vif["VIF"] = [variance_inflation_factor(
        X_df.values, i) for i in range(X_df.shape[1])]
    print('\n--- VIF ---')
    print(vif.to_string())

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.33, random_state=12)

    # LR Model
    model = LogisticRegression(class_weight='balanced', solver='liblinear')
    model.fit(X_train, y_train)

# X_test = StandardScaler().fit_transform(X_test)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

# ROC AUC
    print('\n--- ROC AUC ---')
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    print(f'{roc_auc:.0%}')

# confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('\n--- Confusion Matrix ---')
    tn, fp, fn, tp = cm.ravel()
    print(f'''TN: {tn} FP: {fp}
    FN: {fn} TP: {tp}''')

# Scores
    print('\n--- Scores ---')
    print(f'Specificity: {tn / (tn + fp):.0%}')
    print(f'Precision:   {precision_score(y_test, y_pred):.0%}')
    print(f'Recall:      {recall_score(y_test, y_pred):.0%}')
    print(f'F1:          {f1_score(y_test, y_pred):.0%}')

# Parameters
    print('\n--- Parameters ---')
    print(f'Intercept: {model.intercept_}')
    print(f'Coefficients: {model.coef_}')

    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def lr_pred(intercept, coeffs, features):
        prob_fill = sigmoid(
            intercept + np.sum(np.array(coeffs) * np.array(features)))
        return [1 - prob_fill, prob_fill]

    print('\n--- Sample Data ---')
    print(X_test[0])

    print('\n--- Manual LR on Sample Data ---')
    print(lr_pred(model.intercept_, model.coef_, X_test[0]))

    print('\n--- Actual LR on Sample Data ---')
    print(y_pred_proba[0])
    print(y_pred[0])


if __name__ == '__main__':
    main()
