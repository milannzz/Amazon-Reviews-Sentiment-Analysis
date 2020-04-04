import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_json('sentiment/Books_small.json', lines=True)

ReView = dataset[['reviewText','overall']]

sentiment = []
for i,row in ReView.iterrows():
    if row['overall'] >3:
        sentiment.append("POSITIVE")
    elif row['overall'] <3:
        sentiment.append("NEGATIVE")
    else:
        sentiment.append("NEUTRAL")

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', ReView['reviewText'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

review = review.drop("overall",1)
review = review['reviewText'].astype(str).tolist()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(review , sentiment,test_size = .33 ,random_state = 42) 

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train).toarray()
        ]
X_test_vectors = vectorizer.transform(X_test).toarray()

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy')
classifier.fit(X_train_vectors, y_train)

y_pred = classifier.predict(X_test_vectors).tolist()

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#accuracy
from sklearn.metrics import accuracy_score
print('Accuracy Score :',accuracy_score(y_test, y_pred))
