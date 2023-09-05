import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

import warnings
warnings.simplefilter("ignore")

# Loading the dataset
data = pd.read_csv("deployment-yt/language_detection.csv")



X = data["Text"]
y = data["Language"]

le = LabelEncoder()
y = le.fit_transform(y)

data_list = []
for text in X:
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', ' ', text)
    text = re.sub(r'[[]]', ' ', text)
    text = text.lower()
    data_list.append(text)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
cv = CountVectorizer()
cv.fit(X_train)

x_train = cv.transform(X_train).toarray()
x_test  = cv.transform(X_test).toarray()

model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("Accuracy is :",ac)

pipe = Pipeline([('vectorizer', cv), ('multinomialNB', model)])
pipe.fit(X_train, y_train)

y_pred2 = pipe.predict(X_test)
ac2 = accuracy_score(y_test, y_pred2)
print("Accuracy is :",ac2)

with open('trained_pipeline-0.1.0.pkl','wb') as f:
    pickle.dump(pipe, f)

text = "Hello, how are you?"
text = "Ciao, come stai?"
#text = "Hi, how are you?"
y = pipe.predict([text])
print(le.classes_[y[0]], y)