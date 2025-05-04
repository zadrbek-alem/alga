import pandas as pd
import numpy as np
import torch

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

# ==================== Настройки ====================
seed_everything(42)  # для воспроизводимости

# Используем CUDA если есть
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Используем устройство: {device}")

# ==================== Загрузка данных ====================
# train.csv: day,value
# test.csv: day

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Добавляем нужные столбцы
train_df["time_idx"] = pd.factorize(train_df["day"])[0]
train_df["group"] = "series"  # одна серия — одна группа

# Тесту тоже добавляем group и запомним time_idx
test_df["time_idx"] = pd.factorize(test_df["day"], sort=True)[0] + train_df["time_idx"].max() + 1
test_df["group"] = "series"

# Сливаем всё вместе — это нужно для правильной генерации датасета
all_df = pd.concat([train_df, test_df], sort=False).reset_index(drop=True)

# ==================== Создание TimeSeriesDataSet ====================
max_encoder_length = 30  # сколько дней назад модель "смотрит"
max_prediction_length = len(test_df)  # сколько дней нужно предсказать

# Создаём обучающий датасет
training = TimeSeriesDataSet(
    train_df,
    time_idx="time_idx",
    target="value",
    group_ids=["group"],
    max_encoder_length=max_encoder_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["group"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_reals=["value"],
    target_normalizer=NaNLabelEncoder(),
    add_relative_time_idx=True,
    add_target_scales=True,
)

# Тестовый датасет (без target)
test_dataset = TimeSeriesDataSet.from_dataset(training, all_df, predict=True, stop_randomization=True)

# ==================== DataLoaders ====================
train_dataloader = training.to_dataloader(train=True, batch_size=64)
test_dataloader = test_dataset.to_dataloader(train=False, batch_size=64)

# ==================== Модель ====================
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    loss=torch.nn.MSELoss(),
    log_interval=10,
    reduce_on_plateau_patience=4,
).to(device)

# ==================== Тренировка ====================
trainer = Trainer(
    max_epochs=30,
    gradient_clip_val=0.1,
    accelerator="gpu" if device == "cuda" else "cpu",
    devices=1
)

trainer.fit(tft, train_dataloaders=train_dataloader)

# ==================== Предсказание ====================
predictions = tft.predict(test_dataloader)
print("Предсказания на тесте:")
print(predictions)

# ==================== Сохранение ====================
# Складываем с днями
test_df["predicted_value"] = predictions.numpy()
test_df[["day", "predicted_value"]].to_csv("submission.csv", index=False)
print("Готово. Предсказания сохранены в submission.csv")
