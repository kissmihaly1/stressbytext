import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import nltk
from nltk.corpus import stopwords
import ssl






def Run():

    data = pd.read_csv("data/Stress.csv")
    stemmer = nltk.SnowballStemmer("english")
    stopword = set(stopwords.words('english'))

    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download('stopwords')


    def clean(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = [word for word in text.split(' ') if word not in stopword]
        text = " ".join(text)
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
        return text

    data["text"] = data["text"].apply(clean)

    data["new_label"] = data["label"].map({0: "No Stress", 1: "Stress"})
    data = data[["text", "new_label"]]

    #teach

    x = np.array(data["text"])
    y = np.array(data["new_label"])
    cv = CountVectorizer()
    X = cv.fit_transform(x)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y,
                                                    test_size=0.33,
                                                    random_state=42)

    from sklearn.naive_bayes import BernoulliNB
    model = BernoulliNB()
    model.fit(xtrain, ytrain)

    ypred = model.predict(xtest)

    print(classification_report(ytest, ypred))

    user = input("Text: ")
    data = cv.transform([user]).toarray()
    output = model.predict(data)
    print(output)


Run()