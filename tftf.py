import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Загрузка данных
train = pd.read_csv('train.csv')  # для возможной дообработки или обучения TF-IDF
test_desc = [...]  # список из 10 описаний (первая группа)
test_conc = [...]  # список из 10 заключений (первая группа)

# Объединяем все тексты для обучения TF-IDF
all_texts = train['description'].tolist() + train['conclusion'].tolist() + test_desc + test_conc

# Создаем TF-IDF векторизатор
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
vectorizer.fit(all_texts)

# Векторизуем описание и заключения теста
desc_vecs = vectorizer.transform(test_desc)
conc_vecs = vectorizer.transform(test_conc)

# Считаем матрицу косинусных расстояний (10x10)
similarity_matrix = cosine_similarity(desc_vecs, conc_vecs)

# Находим наиболее вероятные пары — можно использовать жадный метод:
matches = similarity_matrix.argmax(axis=1)

# Печать результатов
for i, j in enumerate(matches):
    print(f"Описание {i} → Заключение {j} (сходство: {similarity_matrix[i, j]:.3f})")
