import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# === 1. Подготовка данных с улучшенной обработкой ===
def prepare_data(train_path, test_path):
    """Улучшенная функция подготовки данных с обработкой выбросов и расширенным преобразованием дат"""
    # Загрузка данных
    train = pd.read_csv(train_path, parse_dates=["submitted_date"])
    test = pd.read_csv(test_path, parse_dates=["week_start", "week_end"])
    
    # Преобразование даты в еженедельный формат
    train["week_start"] = train["submitted_date"] - pd.to_timedelta(train["submitted_date"].dt.weekday, unit="d")
    
    # Агрегация по неделям
    weekly = train.groupby(["category", "week_start"])["num_papers"].sum().reset_index()
    
    # Расширенные временные признаки
    weekly["weekofyear"] = weekly["week_start"].dt.isocalendar().week
    weekly["year"] = weekly["week_start"].dt.year
    weekly["month"] = weekly["week_start"].dt.month
    weekly["quarter"] = weekly["week_start"].dt.quarter
    weekly["dayofyear"] = weekly["week_start"].dt.dayofyear
    
    # Признаки для учета долгосрочных трендов
    weekly["days_since_start"] = (weekly["week_start"] - weekly["week_start"].min()).dt.days
    
    # Праздничные периоды (приблизительно)
    weekly["is_holiday_season"] = ((weekly["month"] == 12) & (weekly["week_start"].dt.day >= 15)) | (weekly["month"] == 1) & (weekly["week_start"].dt.day <= 15)
    weekly["is_summer"] = (weekly["month"] >= 6) & (weekly["month"] <= 8)
    
    # Циклические признаки для недели года и месяца
    weekly["sin_week"] = np.sin(2 * np.pi * weekly["weekofyear"] / 52)
    weekly["cos_week"] = np.cos(2 * np.pi * weekly["weekofyear"] / 52)
    weekly["sin_month"] = np.sin(2 * np.pi * weekly["month"] / 12)
    weekly["cos_month"] = np.cos(2 * np.pi * weekly["month"] / 12)
    
    # Добавление тех же временных признаков к тестовым данным
    test["weekofyear"] = test["week_start"].dt.isocalendar().week
    test["year"] = test["week_start"].dt.year
    test["month"] = test["week_start"].dt.month
    test["quarter"] = test["week_start"].dt.quarter
    test["dayofyear"] = test["week_start"].dt.dayofyear
    test["days_since_start"] = (test["week_start"] - pd.to_datetime("2000-01-01")).dt.days
    test["is_holiday_season"] = ((test["month"] == 12) & (test["week_start"].dt.day >= 15)) | (test["month"] == 1) & (test["week_start"].dt.day <= 15)
    test["is_summer"] = (test["month"] >= 6) & (test["month"] <= 8)
    test["sin_week"] = np.sin(2 * np.pi * test["weekofyear"] / 52)
    test["cos_week"] = np.cos(2 * np.pi * test["weekofyear"] / 52)
    test["sin_month"] = np.sin(2 * np.pi * test["month"] / 12)
    test["cos_month"] = np.cos(2 * np.pi * test["month"] / 12)
    
    return weekly.sort_values(["category", "week_start"]), test

# === 2. Расширенный набор признаков с оптимизированными функциями ===
def create_features(df, lags, roll_windows, ewm_alphas=[0.3, 0.5, 0.7]):
    """Создание комплексного набора признаков, включая лаги, скользящие средние и экспоненциальные скользящие средние"""
    # Базовые лаговые признаки
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("category")["num_papers"].shift(lag)
    
    # Скользящие средние
    for window in roll_windows:
        df[f"rolling_mean_{window}"] = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).mean()
        df[f"rolling_std_{window}"] = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).std()
        df[f"rolling_min_{window}"] = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).min()
        df[f"rolling_max_{window}"] = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).max()
        df[f"rolling_median_{window}"] = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).median()
    
    # Экспоненциальные скользящие средние для учета трендов с разной "памятью"
    for alpha in ewm_alphas:
        df[f"ewm_{alpha}"] = df.groupby("category")["num_papers"].shift(1).ewm(alpha=alpha, min_periods=1).mean()
    
    # Признаки разности для выявления трендов
    for lag in range(1, min(5, max(lags))):
        df[f"diff_{lag}"] = df["num_papers"].diff(lag)
    
    # Признаки роста/падения
    df["growth_1"] = df["num_papers"] / df["lag_1"] - 1
    
    # Скользящее отклонение от среднего для выявления аномалий
    for window in [4, 8, 12]:
        roll_mean = df.groupby("category")["num_papers"].shift(1).rolling(window, min_periods=1).mean()
        df[f"dev_from_mean_{window}"] = df["num_papers"].shift(1) - roll_mean
    
    # Статистика за последний год (52 недели)
    yearly_mean = df.groupby("category")["num_papers"].shift(1).rolling(52, min_periods=1).mean()
    yearly_std = df.groupby("category")["num_papers"].shift(1).rolling(52, min_periods=1).std()
    df["z_score_yearly"] = (df["num_papers"].shift(1) - yearly_mean) / yearly_std.replace(0, 1)
    
    # Подсчет сверхактивных недель (пики)
    threshold = df.groupby("category")["num_papers"].shift(1).rolling(12, min_periods=1).quantile(0.8)
    df["is_peak_week"] = (df["num_papers"].shift(1) > threshold).astype(int)
    
    # Признаки, основанные на номере недели в году
    for cat in df["category"].unique():
        cat_df = df[df["category"] == cat]
        week_means = cat_df.groupby("weekofyear")["num_papers"].mean().to_dict()
        week_stds = cat_df.groupby("weekofyear")["num_papers"].std().fillna(0).to_dict()
        df.loc[df["category"] == cat, "week_mean"] = df.loc[df["category"] == cat, "weekofyear"].map(week_means)
        df.loc[df["category"] == cat, "week_std"] = df.loc[df["category"] == cat, "weekofyear"].map(week_stds)
    
    return df

# === 3. Улучшенная кросс-валидация с учетом временных рядов ===
def validate_model(df, cat, features, target="num_papers", n_splits=3, forecast_horizon=8):
    """Кросс-валидация модели с учетом временной структуры данных"""
    cat_df = df[df["category"] == cat].copy()
    
    # Очистка признаков и подготовка данных
    cat_df = cat_df.replace([np.inf, -np.inf], np.nan)
    cat_df = cat_df.fillna(method='ffill').fillna(method='bfill')
    
    # Временная кросс-валидация
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=forecast_horizon)
    
    # Используем RobustScaler для устойчивости к выбросам
    scaler = RobustScaler()
    
    fold_errors = []
    fold_models = []
    
    # Параметры модели LightGBM, оптимизированные для временных рядов
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'metric': 'mape',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': -1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': 5,
        'lambda_l1': 0.5,
        'lambda_l2': 0.5,
        'verbose': -1
    }
    
    X = cat_df[features].values
    y = cat_df[target].values
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Масштабирование признаков
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение модели
        lgb_train = lgb.Dataset(X_train_scaled, label=y_train)
        lgb_valid = lgb.Dataset(X_test_scaled, label=y_test)
        
        model = lgb.train(
            params,
            lgb_train,
            num_boost_round=3000,
            valid_sets=[lgb_valid],
            callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
        )
        
        # Сохранение модели
        fold_models.append((model, scaler))
        
        # Оценка ошибки
        y_pred = model.predict(X_test_scaled)
        
        # Расчет Safe MAPE (предотвращает деление на очень маленькие значения)
        denominator = np.maximum(np.abs(y_test), 10.0)
        mape = np.mean(np.abs(y_pred - y_test) / denominator)
        fold_errors.append(mape)
    
    # Выбор лучшей модели по средней ошибке
    best_model_idx = np.argmin(fold_errors)
    best_model, best_scaler = fold_models[best_model_idx]
    
    return best_model, best_scaler, np.mean(fold_errors)

# === 4. Функция для итеративного прогнозирования с корректировкой ===
def predict_category(cat_df, test_df, features, forecast_horizon=8):
    """Улучшенный прогноз с корректировкой экстремальных значений"""
    # Отбор последних данных для инициализации прогноза
    recent_data = cat_df.tail(max(features.count(lambda f: 'lag' in f) + 10, 52)).copy()
    cat_test = test_df[test_df["category"] == cat_df["category"].iloc[0]].copy()
    
    # Обучение модели с кросс-валидацией
    model, scaler, cv_error = validate_model(cat_df, cat_df["category"].iloc[0], features)
    
    # История прогнозов
    predictions = []
    
    # Итеративное прогнозирование
    for i in range(len(cat_test)):
        row = cat_test.iloc[i]
        week_start = row["week_start"]
        
        # Получение всех необходимых признаков
        test_record = pd.DataFrame([{
            col: row[col] for col in cat_test.columns if col in features
        }])
        
        # Добавление всех лаговых и скользящих признаков
        for col in features:
            if col not in test_record.columns:
                if "lag_" in col and int(col.split("_")[1]) <= len(recent_data):
                    lag_value = int(col.split("_")[1])
                    test_record[col] = recent_data["num_papers"].iloc[-lag_value]
                elif "rolling_" in col:
                    window = int(col.split("_")[-1])
                    func = col.split("_")[1]
                    if func == "mean":
                        test_record[col] = recent_data["num_papers"].tail(window).mean()
                    elif func == "std":
                        test_record[col] = recent_data["num_papers"].tail(window).std()
                    elif func == "min":
                        test_record[col] = recent_data["num_papers"].tail(window).min()
                    elif func == "max":
                        test_record[col] = recent_data["num_papers"].tail(window).max()
                    elif func == "median":
                        test_record[col] = recent_data["num_papers"].tail(window).median()
                elif "ewm_" in col:
                    alpha = float(col.split("_")[1])
                    test_record[col] = recent_data["num_papers"].ewm(alpha=alpha).mean().iloc[-1]
                elif "diff_" in col:
                    lag = int(col.split("_")[1])
                    if len(recent_data) >= lag:
                        test_record[col] = recent_data["num_papers"].iloc[-1] - recent_data["num_papers"].iloc[-1-lag]
                elif col == "growth_1":
                    test_record[col] = recent_data["num_papers"].iloc[-1] / recent_data["num_papers"].iloc[-2] - 1 if recent_data["num_papers"].iloc[-2] > 0 else 0
                elif "dev_from_mean_" in col:
                    window = int(col.split("_")[-1])
                    test_record[col] = recent_data["num_papers"].iloc[-1] - recent_data["num_papers"].tail(window).mean()
                elif col == "z_score_yearly":
                    yearly_mean = recent_data["num_papers"].tail(min(52, len(recent_data))).mean()
                    yearly_std = recent_data["num_papers"].tail(min(52, len(recent_data))).std()
                    test_record[col] = (recent_data["num_papers"].iloc[-1] - yearly_mean) / yearly_std if yearly_std > 0 else 0
                elif col == "is_peak_week":
                    threshold = recent_data["num_papers"].tail(min(12, len(recent_data))).quantile(0.8)
                    test_record[col] = 1 if recent_data["num_papers"].iloc[-1] > threshold else 0
                elif col == "week_mean":
                    week_num = row["weekofyear"]
                    same_weeks = cat_df[cat_df["weekofyear"] == week_num]["num_papers"]
                    test_record[col] = same_weeks.mean() if not same_weeks.empty else recent_data["num_papers"].mean()
                elif col == "week_std":
                    week_num = row["weekofyear"]
                    same_weeks = cat_df[cat_df["weekofyear"] == week_num]["num_papers"]
                    test_record[col] = same_weeks.std() if not same_weeks.empty and len(same_weeks) > 1 else recent_data["num_papers"].std()
        
        # Масштабирование признаков
        test_features = test_record[features].fillna(0)
        test_features = test_features.replace([np.inf, -np.inf], 0)
        test_features_scaled = scaler.transform(test_features)
        
        # Прогноз
        prediction = model.predict(test_features_scaled)[0]
        
        # Корректировка экстремальных значений на основе исторических данных
        category_min = max(0, cat_df["num_papers"].min() * 0.9)  # Не ниже 0 или 90% от минимума
        category_max = cat_df["num_papers"].max() * 1.1  # Не выше 110% от максимума
        
        # Особая обработка для недель с историческими аномалиями
        current_week = row["weekofyear"]
        historical_same_week = cat_df[cat_df["weekofyear"] == current_week]["num_papers"]
        
        if not historical_same_week.empty:
            week_mean = historical_same_week.mean()
            week_std = historical_same_week.std()
            
            # Если прогноз выходит за пределы исторического диапазона для этой недели года
            if prediction > week_mean + 3 * week_std:
                prediction = min(prediction, week_mean + 3 * week_std)
            elif prediction < week_mean - 2 * week_std:
                prediction = max(prediction, week_mean - 2 * week_std)
        
        # Применение общих ограничений
        prediction = max(category_min, min(category_max, prediction))
        
        # Округление предсказания до целого числа, так как количество статей не может быть дробным
        prediction = round(prediction)
        
        # Сохранение прогноза
        predictions.append(prediction)
        
        # Обновление истории для следующей итерации
        recent_data = pd.concat([
            recent_data,
            pd.DataFrame([{
                "category": row["category"],
                "week_start": row["week_start"],
                "num_papers": prediction,
                "weekofyear": row["weekofyear"],
                "year": row["year"],
                "month": row["month"],
                "quarter": row["quarter"],
                "dayofyear": row["dayofyear"],
                "days_since_start": row["days_since_start"],
                "is_holiday_season": row["is_holiday_season"],
                "is_summer": row["is_summer"],
                "sin_week": row["sin_week"],
                "cos_week": row["cos_week"],
                "sin_month": row["sin_month"],
                "cos_month": row["cos_month"]
            }])
        ], ignore_index=True)
        
        # Обновление признаков для следующей итерации
        recent_data = create_features(
            recent_data, 
            lags=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 26, 52], 
            roll_windows=[2, 4, 8, 12, 16, 26, 52],
            ewm_alphas=[0.3, 0.5, 0.7]
        )
    
    cat_test["num_papers"] = predictions
    return cat_test

# === 5. Визуализация прогнозов для анализа ===
def plot_predictions(category, train_data, test_data):
    """Функция для визуализации исторических данных и прогноза"""
    plt.figure(figsize=(12, 6))
    
    # Исторические данные
    cat_train = train_data[train_data["category"] == category]
    plt.plot(cat_train["week_start"], cat_train["num_papers"], marker='o', label='Исторические данные')
    
    # Прогноз
    cat_test = test_data[test_data["category"] == category]
    plt.plot(cat_test["week_start"], cat_test["num_papers"], marker='*', color='red', label='Прогноз')
    
    plt.title(f'Прогноз количества статей для категории {category}')
    plt.xlabel('Дата')
    plt.ylabel('Количество статей')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Сохранение графика
    plt.savefig(f"prediction_{category.replace('/', '_')}.png")
    plt.close()

# === 6. Основная функция ===
def main(train_path="/kaggle/input/kazakhstan-ai-respa-take-home/train.csv", test_path="/kaggle/input/kazakhstan-ai-respa-take-home/test.csv"):
    """Основная функция для запуска процесса прогнозирования"""
    print("Загрузка и подготовка данных...")
    weekly, test = prepare_data(train_path, test_path)
    
    # Определение признаков
    basic_features = [
        "weekofyear", "year", "month", "quarter", "dayofyear", 
        "days_since_start", "is_holiday_season", "is_summer",
        "sin_week", "cos_week", "sin_month", "cos_month"
    ]
    
    # Создание расширенных признаков
    print("Создание признаков...")
    weekly = create_features(
        weekly, 
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 26, 52], 
        roll_windows=[2, 4, 8, 12, 16, 26, 52],
        ewm_alphas=[0.3, 0.5, 0.7]
    )
    
    # Получение списка всех созданных признаков
    all_features = weekly.columns.difference(["category", "week_start", "num_papers"])
    
    # Очистка признаков от пропущенных значений
    weekly = weekly.replace([np.inf, -np.inf], np.nan)
    weekly = weekly.fillna(method='ffill').fillna(method='bfill')
    
    # Прогнозирование для каждой категории
    print("Начало прогнозирования по категориям...")
    all_predictions = []
    
    for cat in test["category"].unique():
        print(f"Обработка категории: {cat}")
        cat_data = weekly[weekly["category"] == cat].copy()
        
        if len(cat_data) < 10:  # Слишком мало данных для надежного прогноза
            print(f"Слишком мало данных для категории {cat}, использование базового метода")
            cat_test = test[test["category"] == cat].copy()
            # Базовый метод для категорий с малым количеством данных
            if len(cat_data) > 0:
                cat_test["num_papers"] = int(cat_data["num_papers"].mean())
            else:
                cat_test["num_papers"] = int(weekly["num_papers"].mean())  # Среднее по всем категориям
            all_predictions.append(cat_test)
            continue
        
        # Выбор наиболее важных признаков для данной категории
        features_to_use = list(all_features)
        
        # Прогнозирование
        cat_predictions = predict_category(cat_data, test, features_to_use)
        all_predictions.append(cat_predictions)
        
        # Визуализация прогноза (опционально)
        try:
            plot_predictions(cat, weekly, cat_predictions)
        except Exception as e:
            print(f"Ошибка при визуализации для {cat}: {e}")
    
    # Сборка финальных предсказаний
    print("Формирование итогового submission файла...")
    submission = pd.concat(all_predictions)
    submission["id"] = submission["category"] + "__" + submission["week_id"].astype(str)
    
    # Сохранение результатов
    submission[["id", "num_papers"]].to_csv("submission.csv", index=False)
    print("Готово! Файл submission.csv успешно создан.")

if __name__ == "__main__":
    main()
