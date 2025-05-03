import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import ComplementNB

def preprocess(text):
    return text.lower()

def extract_features(text_list):
    processed_texts = [preprocess(t) for t in text_list]
    return vectorizer.transform(processed_texts)

def read_data_known(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df = df[df['category'] != 'UNKNOWN']
    N = len(df)
    df_train = df.iloc[:N // 2]
    df_test = df.iloc[N // 2:]
    return df_train['text'].tolist(), df_train['category'].str.strip().tolist(), df_test['text'].tolist(), df_test['category'].str.strip().tolist()

def read_data(file_path):
    df = pd.read_csv(file_path, sep='\t')
    df_train = df[df['category'] != 'UNKNOWN']
    df_test = df[df['category'] == 'UNKNOWN']
    return df_train['text'].tolist(), df_train['category'].str.strip().tolist(), df_test['text'].tolist()

MODE = "SUBMIT" 

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    stop_words='english',
    max_features=1000000,
    sublinear_tf=True
)

if MODE == "LOCAL":
    X_train, y_train, X_test, y_test = read_data_known("input.txt")
else:
    X_train, y_train, X_test = read_data("input.txt")
    y_test = None

X_train_t = vectorizer.fit_transform(X_train)
X_test_t = vectorizer.transform(X_test)

model = ComplementNB(alpha=0.4, norm=True)
model.fit(X_train_t, y_train)
y_predict = model.predict(X_test_t)

if y_test:
    correct = (np.array(y_test) == np.array(y_predict)).sum()
    accuracy = correct / len(y_test) * 100
    print("ACCURACY:", accuracy)

with open("output.txt", "w") as f:
    f.write("\n".join(y_predict) + "\n")
