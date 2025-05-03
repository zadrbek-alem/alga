import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('train.csv', sep="\t")
test_data = pd.read_csv('test.csv', sep="\t")

def f(text):
    text = text.lower()
    return text

train_data["Text"] = train_data["Text"].apply(f)
test_data["Text"] = test_data["Text"].apply(f)

vectorizer = TfidfVectorizer(max_features=10000)
X_train = vectorizer.fit_transform(train_data["Text"])
y_train = np.where(train_data["Score"] == "Positive", 1, 0)

X_test = vectorizer.transform(test_data["Text"])

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

submission = pd.DataFrame({
    "idx": test_data["idx"],
    "Score": np.where(y_pred == 1, "Positive", "Negative")
})

submission.to_csv('submission.csv', index=False, sep="\t")
