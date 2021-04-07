import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

conn = psycopg2.connect(
    host='',
    database='',
    user='',
    password='')

cursor = conn.cursor()
cursor.itersize = 10
cursor.execute(
    'SELECT * FROM orders_ml WHERE exchange = \'Bitmex\' and quantity > 0')

bitmex_buy = []
bitmex_sell = []
deribit_buy = []
deribit_sell = []


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

col_names = []
for row in bitmex_buy:
    features = row[3]
    submit_timestamp = row[4]
    is_filled = row[5]

    contains_inf_or_nan = False
    values = []

    for feature in features:
        name = feature['name']
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

# Scale training data
# X_train = StandardScaler().fit_transform(X_train)

# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Feature importance
print('\n--- Importance ---')
# print(model.feature_importances_)
feature_importance = zip(col_names, model.feature_importances_)
feature_importance_df = pd.DataFrame(
    feature_importance, columns=['name', 'importance'])
print(feature_importance_df.to_string())

# Predict data
# X_test = StandardScaler().fit_transform(X_test)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# ROC AUC
print('\n--- ROC AUC ---')
print(roc_auc_score(y_test, y_pred_proba[:, 1]))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('\n--- Confusion Matrix ---')
print(cm)
tn, fp, fn, tp = cm.ravel()

# Scores
print('\n--- Scores ---')
print(f'Specificity: {tn / (tn + fp)}')
print(f'Precision: {precision_score(y_test, y_pred)}')
print(f'Recall: {recall_score(y_test, y_pred)}')
print(f'F1: {f1_score(y_test, y_pred)}')
