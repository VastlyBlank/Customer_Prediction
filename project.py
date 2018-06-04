# The below uses example provided by
# https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44

import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

# Settings to be able to print all the columns of data set
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)

# Import data file. testdata.csv was scrubbed to remove StockCodes with letters, negative quantities,
# and blank descriptions.
df = pd.read_csv('testdata.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})

dataset = df[["ChristmasBag", "StockCode", "Description", "Quantity", "UnitPrice", "CustomerID"]].dropna(axis=0, how='any')

print(len(dataset))

purchased = []
for i in range(len(dataset)):
    purchased.append(1)

# Data set of how much of each item each customers purchased.
# Note that the ChristmasBag field is the count of each item.
customerDataset = dataset[["ChristmasBag", "StockCode", "CustomerID"]]
customerDataset["PurchasedCount"] = purchased
customerDataset = customerDataset.groupby(["CustomerID", "StockCode"]).count()
print(customerDataset)

# Reduced data set size for faster testing, will expand to full data set eventually
dataset = dataset[0:]

# Splits data into training and test sets
X_train, X_test = train_test_split(dataset, test_size=0.5, random_state=int(time.time()))

# Set up bayes classifier
gnb = GaussianNB()
used_features = ["ChristmasBag", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
gnb.fit(X_train[used_features].values, X_train["ChristmasBag"])
y_pred = gnb.predict(X_test[used_features])

print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
          X_test.shape[0], (X_test["ChristmasBag"] != y_pred).sum(),
          100*(1-(X_test["ChristmasBag"] != y_pred).sum()/X_test.shape[0])
))

mean_christmas_bag = np.mean(X_train["ChristmasBag"])
mean_not_christmas_bag = 1 - mean_christmas_bag
print("\nChristmas Bag purchased prob = {:03.2f}%, Not purchased prob = {:03.2f}%"
      .format(100*mean_christmas_bag, 100*mean_not_christmas_bag))

mean_products_purchased = np.mean(X_train[X_train["ChristmasBag"] == 1]["StockCode"].unique())
std_products_purchased = np.std(X_train[X_train["ChristmasBag"] == 1]["StockCode"].count())
mean_products_not_purchased = np.mean(X_train[X_train["ChristmasBag"] == 0]["StockCode"].count())
std_products_not_purchased = np.std(X_train[X_train["ChristmasBag"] == 0]["StockCode"].count())

print("\nmean_products_purchased = {:03.2f}".format(mean_products_purchased))
print("std_products_purchased = {:03.2f}".format(std_products_purchased))
print("mean_products_not_purchased = {:03.2f}".format(mean_products_not_purchased))
print("std_products_not_purchased = {:03.2f}".format(std_products_not_purchased ))