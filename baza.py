# ===============================
# 📦 Импорт библиотек
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')
sns.set(style='whitegrid')

# ===============================
# 📊 Загрузка и первичный осмотр данных
# ===============================
# df = pd.read_csv('your_dataset.csv')
# Пример для шаблона:
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randint(0, 5, 100),
    'target': np.random.randint(0, 2, 100)
})

print(df.head())
print(df.describe())
print(df.info())

# ===============================
# 🔍 Быстрый EDA
# ===============================
for col in df.select_dtypes(include='number').columns:
    plt.figure(figsize=(6, 3))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.show()

sns.pairplot(df, hue='target')
plt.show()

# ===============================
# 🧹 Препроцессинг
# ===============================
X = df.drop('target', axis=1)
y = df['target']

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ===============================
# ✂️ Train/Val Split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# ===============================
# 🔁 Кросс-валидация и модель
# ===============================
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1')

print("CV F1 Scores:", cv_scores)
print("Mean F1:", np.mean(cv_scores))

# ===============================
# 🧪 Финальная валидация
# ===============================
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))
print("ROC AUC:", roc_auc_score(y_val, y_pred))
