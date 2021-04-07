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

conn = psycopg2.connect(
    host='',
    database='',
    user='',
    password='')

cursor = conn.cursor()
cursor.itersize = 10
cursor.execute(
    'SELECT * FROM orders_ml')

bitmex_buy = []
bitmex_sell = []
deribit_buy = []
deribit_sell = []
exchanges_sides = [bitmex_buy, bitmex_sell, deribit_buy, deribit_sell]


for row in cursor:
    exchange = row[1]
    is_buy = row[2] > 0

    if exchange == 'Bitmex' and is_buy:
        bitmex_buy.append(row)
    elif exchange == 'Bitmex' and not is_buy:
        bitmex_sell.append(row)
    elif exchange == 'Deribit' and is_buy:
        deribit_buy.append(row)
    elif exchange == 'Deribit' and not is_buy:
        deribit_sell.append(row)

X = []
y = []

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

for exchange_index, exchange_side in enumerate(exchanges_sides):
    col_names = []

    print(f'\n === Processing Exchange Index: {exchange_index} ===')

    for row in exchange_side:
        features = row[3]
        submit_timestamp = row[4]
        is_filled = row[5]

        contains_inf_or_nan = False
        values = []

        for feature in features:
            name = feature['name']
            if name not in chosen_features:
                continue

            value = feature['value']

            if name not in col_names:
                col_names.append(name)

            if value is None or np.isnan(value) or np.isinf(value):
                contains_inf_or_nan = True
                break

            values.append(value)

        if contains_inf_or_nan:
            continue

        X.append(values)
        y.append(is_filled)

    X_df = pd.DataFrame(X, columns=col_names)

# Multicollinearity - VIF
    vif = pd.DataFrame()
    vif['features'] = X_df.columns
    vif["VIF"] = [variance_inflation_factor(
        X_df.values, i) for i in range(X_df.shape[1])]
    print('\n--- VIF ---')
    print(vif.to_string())

# Split train/test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=12)

# Scaler
# X_train = StandardScaler().fit_transform(X_train)

# Model
    model = LogisticRegression(class_weight='balanced', solver='liblinear')

# Grid Search
# model = GridSearchCV(model, param_grid={'penalty': ['l1', 'l2'], 'C': [
#     0.001, .009, 0.01, .09, 1, 5, 10, 25]},
#     scoring='roc_auc')
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


# Plot

# calculate scores
# ns_probs = [0 for _ in range(len(y_test))]
# ns_auc = roc_auc_score(y_test, ns_probs)
# lr_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
# # summarize scores
# print('No Skill: ROC AUC=%.3f' % (ns_auc))
# print('Logistic: ROC AUC=%.3f' % (lr_auc))
# # calculate roc curves
# ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs)
# lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
# # plot the roc curve for the model
# pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
# pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
# # axis labels
# pyplot.xlabel('False Positive Rate')
# pyplot.ylabel('True Positive Rate')
# # show the legend
# pyplot.legend()
# # show the plot
# pyplot.show()


# calculate precision-recall curve
# lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_pred_proba[:, 1])
# lr_f1, lr_auc = f1_score(y_test, y_pred), auc(lr_recall, lr_precision)
# # summarize scores
# print('Logistic: f1=%.3f auc=%.3f' % (lr_f1, lr_auc))
# # plot the precision-recall curves
# no_skill = np.count_nonzero(y_test) / len(y_test)
# pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# pyplot.plot(lr_recall, lr_precision, marker='.', label='Logistic')
# # axis labels
# pyplot.xlabel('Recall')
# pyplot.ylabel('Precision')
# # show the legend
# pyplot.legend()
# # show the plot
# pyplot.show()
