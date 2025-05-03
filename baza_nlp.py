# ===============================
# 📦 Импорт библиотек
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re
import string

# ===============================
# 🧹 Текстовая предобработка
# ===============================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

# ===============================
# 📊 Загрузка данных
# ===============================
# Пример для шаблона
df = pd.DataFrame({
    'text': [
        'This is a good movie!',
        'Terrible acting and bad plot.',
        'I loved the visuals and music.',
        'Worst movie ever made.',
        'An average film, nothing special.'
    ],
    'label': [1, 0, 1, 0, 0]
})

df['clean_text'] = df['text'].apply(clean_text)

# ===============================
# ✂️ Train/Val Split
# ===============================
X_train, X_val, y_train, y_val = train_test_split(
    df['clean_text'], df['label'], test_size=0.2, random_state=42)

# ===============================
# 🧠 TF-IDF + Logistic Regression
# ===============================
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)

model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_val_vec)

print("Accuracy:", accuracy_score(y_val, y_pred))
print("F1 Score:", f1_score(y_val, y_pred))

# ===============================
# 🔁 HuggingFace Transformers (BERT)
# ===============================
# Можно раскомментировать если нужно:
"""
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(X_train.tolist(), y_train.tolist())
val_dataset = TextDataset(X_val.tolist(), y_val.tolist())

model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="epoch",
    save_strategy="no",
    logging_dir="./logs",
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
"""
