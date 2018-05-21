import pandas
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

training_data = "training_data/sms_spam.csv"
data = pandas.read_csv(training_data)

clf = MultinomialNB()
vectorizer = CountVectorizer()

message = vectorizer.fit_transform(data["text"].values)
targets = data["type"].values
clf.fit(message,targets)
print(clf.predict(vectorizer.transform(["hiii,how are u"])))
