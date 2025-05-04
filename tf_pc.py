from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Векторизуем текст
vectorizer = TfidfVectorizer(max_features=10000)
X_tfidf = vectorizer.fit_transform(combined_texts)

# PCA требует плотную матрицу (dense)
X_dense = X_tfidf.toarray()

# Применяем PCA — допустим, оставим 300 компонент
pca = PCA(n_components=300)
X_pca = pca.fit_transform(X_dense)

from sklearn.decomposition import TruncatedSVD

svd = TruncatedSVD(n_components=300)
X_svd = svd.fit_transform(X_tfidf)

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=10000)),
    ('svd', TruncatedSVD(n_components=300)),
    ('clf', LogisticRegression())
])

pipeline.fit(combined_texts, labels)
