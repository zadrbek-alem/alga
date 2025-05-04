import xgboost as xgb
from sklearn.metrics import accuracy_score

# Преобразование в DMatrix (специальный формат XGBoost)
dtrain = xgb.DMatrix(x_train, label=y_train)

# Настройки модели
params = {
    'objective': 'binary:logistic',  # или 'multi:softmax', 'reg:squarederror'
    'eval_metric': 'logloss',
    'verbosity': 0
}

# Обучение модели
bst = xgb.train(params, dtrain, num_boost_round=100)

# Предсказание
y_pred_prob = bst.predict(dtrain)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]

# Оценка
acc = accuracy_score(y_train, y_pred)
print(f"Accuracy: {acc:.4f}")
