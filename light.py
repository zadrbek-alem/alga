import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score  # или другая метрика

# Разделение, если ещё не делал
# x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Создание dataset'а для LightGBM
train_data = lgb.Dataset(x_train, label=y_train)

# Настройки модели
params = {
    'objective': 'binary',         # или 'multiclass', 'regression' — зависит от задачи
    'metric': 'binary_logloss',   # или 'multi_logloss', 'rmse' и др.
    'boosting_type': 'gbdt',
    'verbose': -1
}

# Обучение модели
model = lgb.train(params, train_data, num_boost_round=100)

# Предсказание
y_pred_prob = model.predict(x_train)
y_pred = [1 if prob > 0.5 else 0 for prob in y_pred_prob]  # для бинарной классификации

# Оценка
acc = accuracy_score(y_train, y_pred)
print(f"Accuracy: {acc:.4f}")
