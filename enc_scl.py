from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    MaxAbsScaler,
    RobustScaler,
    Normalizer,
    QuantileTransformer,
    PowerTransformer
)
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder  # только для y
)
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Примерные данные
df = pd.DataFrame({
    'age': [25, 32, 47, 51],
    'salary': [50000, 60000, 80000, 120000],
    'city': ['Moscow', 'SPb', 'Moscow', 'Kazan']
})

y = [0, 1, 0, 1]

# Разделение
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

# Выделение признаков
num_cols = ['age', 'salary']
cat_cols = ['city']

# --- Скейлинг числовых признаков ---
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# --- Кодирование категориальных признаков ---
encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
X_train_cat = encoder.fit_transform(X_train[cat_cols])
X_test_cat = encoder.transform(X_test[cat_cols])

# --- Объединение обработанных признаков ---
import numpy as np
X_train_final = np.hstack([X_train_num, X_train_cat])
X_test_final = np.hstack([X_test_num, X_test_cat])

# Проверка
print("X_train_final shape:", X_train_final.shape)
print("X_test_final shape:", X_test_final.shape)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)


from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# Примерные данные
train_dates = pd.to_datetime(["2021-01-01", "2021-06-01", "2021-12-31"])
test_dates = pd.to_datetime(["2022-01-01", "2022-06-01"])

# Преобразуем даты в числа (например, в timestamp)
train_numeric = train_dates.astype(np.int64).values.reshape(-1, 1)
test_numeric = test_dates.astype(np.int64).values.reshape(-1, 1)

# Нормализация
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_numeric)  # fit только на train
test_scaled = scaler.transform(test_numeric)       # transform на test
