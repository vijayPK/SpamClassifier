# HamOrSpam
This is spam classifier using Naive Bayes classifiers, a family of classifiers that are based on the popular Bayes’
probability theorem. It basicaaly say whether a given message is spam or not. 

# Requirements

sklearn
http://scikit-learn.org/stable/install.html

pandas
https://pandas.pydata.org/pandas-docs/stable/install.html


To know more about how Naive Bayes classifiers work check the link below.
http://sebastianraschka.com/Articles/2014_naive_bayes_1.html


# Working
Here Multinovial Naive Bayes classifier is used.
Functions readFile and readFiles read file and then a dataframe is created using dataFrameFromDirectory methos for training our model. CountVectorizer is used vectorize the data. Then its is feed into classifier for training.

Using predict method of classifier are used to predict whether mail is spam or not. 
