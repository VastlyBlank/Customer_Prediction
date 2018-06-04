import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)

df = pd.read_csv('testdata.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})

dataset = df[["StockCode", "Description", "Quantity", "UnitPrice", "CustomerID"]].dropna(axis=0, how='any')

dataset = dataset[0:100]

X_train, X_test = train_test_split(dataset, test_size=0.5, random_state=int(time.time()))


gnb = GaussianNB()

used_features = ["StockCode", "Quantity", "UnitPrice", "CustomerID"]

gnb.fit(X_train[used_features].values, X_train["CustomerID"])

y_pred = gnb.predict(X_test[used_features])


print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
          X_test.shape[0], (X_test["CustomerID"] != y_pred).sum(),
          100*(1-(X_test["CustomerID"] != y_pred).sum()/X_test.shape[0])
))

