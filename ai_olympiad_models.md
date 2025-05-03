
Тактика на тур олимпиады по ИИ
===============================

🎯 Общая стратегия
------------------
Цель: Быстро получить рабочее решение, а затем улучшать его по ходу соревнования.

🧩 1. Анализ задания (10–20 минут)
---------------------------------
- Прочитай statement.
- Пойми метрику, цель, формат данных, ограничения.
- Запусти baseline (если есть).

🧠 2. Поиск по Kaggle/GitHub (30–60 минут)
-----------------------------------------
- Ищи по ключевым словам.
- Смотри kernels/notebooks.
- Сохраняй интересные архитектуры, подходы, фичи.

📊 3. Быстрый EDA (30–45 минут)
-------------------------------
- Построй распределения, визуализации, корреляции.
- Определи полезные признаки.

⚙️ 4. Baseline модель (1–2 часа)
--------------------------------
- Таблички: LightGBM / CatBoost / XGBoost
- NLP: TF-IDF + LogisticRegression, либо DistilBERT
- CV: EfficientNet, ResNet18, timm

🚀 5. Улучшения (циклично, по 30–60 минут)
-----------------------------------------
- Новые фичи, аугментации
- Улучшенные модели
- Снижение переобучения
- Постобработка, TTA
- Ансамбли

🧠 6. Документация и библиотеки
-------------------------------
- Читай: sklearn, transformers, lightgbm, timm, albumentations, sentence-transformers и др.
- Проверяй копипасту!

📁 7. Организация
-----------------
- Веди чистый код.
- Сохраняй модели, метрики, веса.
- Логируй (wandb/mlflow/tensorboard).

✅ 8. Финальный шаг
-------------------
- Зафиксируй лучшее решение.
- Убедись в воспроизводимости.

Лучшие публичные модели по типам задач
======================================

1. Табличные данные
-------------------
- LightGBM / XGBoost / CatBoost
- TabNet (pytorch-tabnet)
- SAINT / FTTransformer (GitHub)

2. NLP
------
- distilbert-base-uncased
- xlm-roberta-base
- sentence-transformers/all-MiniLM-L6-v2
- bert-base-cased
- facebook/bart-large-cnn
- Helsinki-NLP/opus-mt-en-ru

3. CV
-----
- resnet18, efficientnet_b0 (torchvision, timm)
- convnext_tiny, vit_base_patch16_224 (timm)
- YOLOv5s
- segformer_b0, deeplabv3_resnet50
- clip-vit-base-patch16

4. Time Series
--------------
- tsfresh, prophet, statsmodels
- DLinear, Informer, Autoformer
- LSTM/GRU/RNN

5. Универсальные энкодеры
--------------------------
- sentence-transformers
- CLIP, OpenCLIP
- DINOv2, SAM

Топ-универсальные модели
-------------------------
- NLP: sentence-transformers/all-MiniLM-L6-v2
- CV: timm + efficientnet_b0
- Multimodal: openai/clip-vit-base-patch32
- Таблички: lightgbm + ручные фичи
