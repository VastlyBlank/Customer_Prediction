import pandas as pd
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn import tree

'''Settings to be able to print all the columns of data set'''
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)
pd.options.display.max_rows = 400

'''Import data file. testdata.csv was scrubbed to remove StockCodes with letters, 
negative quantities, and blank descriptions.'''
df = pd.read_csv('testdata2.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
dataset = df[["giftBag", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceMonth",
              "UnitPrice", "CustomerID", "CountryCode"]].dropna(axis=0, how='any')

rawdataset = deepcopy(dataset[["StockCode", "giftBag", "InvoiceNo", "Quantity", "InvoiceMonth", "CountryCode"]])

'''Splits data into training and test sets'''
trainset, testset = train_test_split(rawdataset, test_size=0.2, random_state=12)
#trainset = testset = rawdataset


'''Gaussian Bayes classifier'''
gnb = MultinomialNB()
used_features = ["StockCode"]
gnb.fit(trainset[used_features].values, trainset["CountryCode"])

y_pred = gnb.predict(22802)
print(y_pred)

'''Decision Tree'''
tree_stuff = tree.DecisionTreeClassifier()
tree_stuff.fit(trainset[used_features].values, trainset["CountryCode"])

t_pred = tree_stuff.predict(22802)
print(t_pred)

'''K-NN'''
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(trainset[used_features].values, trainset["CountryCode"])

k_pred = knn.predict(22802)

print(k_pred)


#print(metrics.accuracy_score(testset["CountryCode"], y_pred))
#print(metrics.accuracy_score(testset["CountryCode"], t_pred))
#print(metrics.accuracy_score(testset["CountryCode"], k_pred))
