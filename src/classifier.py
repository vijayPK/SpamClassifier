import os
import io
import numpy
from pandas import DataFrame

from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer 

def readFile(root,filename):
	path = os.path.join(root, filename)
	inBody = False
	lines  = []
	f = io.open(path,'r',encoding = "latin1")
	for line in f:
		if inBody:
			lines.append(line)
		elif line == "\n":
			inBody = True
	f.close()
	return "\n".join(lines)

def readFiles(path):
	for root, dirnames, filenames in os.walk(path):
		for filename in filenames:
			message = readFile(root,filename)
			yield path,message

def dataFrameFromDirectory(path, classification):
	rows  = []
	index = []
	for filename, message in readFiles(path):
		rows.append({"message":message,"class":classification})
		index.append(filename)

	return DataFrame(rows,index = index)

data = DataFrame({"message":[],"class":[]})

data = data.append(dataFrameFromDirectory("/home/imharin/iamHarin/Project/Spam_Classifier/dataset/spam/","spam"))
data = data.append(dataFrameFromDirectory("/home/imharin/iamHarin/Project/Spam_Classifier/dataset/ham/","ham"))

vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data["message"].values)

classifier = MultinomialNB()
targets = data["class"].values
classifier.fit(counts,targets)

# ========Input===============
path = "/home/imharin/iamHarin/Project/Spam_Classifier/dataset/"
filename = "check.eml"


text_count = vectorizer.transform([readFile(path,filename)])
p = classifier.predict(text_count)
print(p)