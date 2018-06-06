# The below uses example provided by
# https://blog.sicara.com/naive-bayes-classifier-sklearn-python-example-tips-42d100429e44

import pandas as pd
import numpy as np
from copy import deepcopy
from collections import Counter
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn import metrics

'''Settings to be able to print all the columns of data set'''
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)
pd.options.display.max_rows = 400

'''Import data file. testdata.csv was scrubbed to remove StockCodes with letters, 
negative quantities, and blank descriptions.'''
df = pd.read_csv('testdata.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
dataset = df[["giftBag", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceMonth",
              "UnitPrice", "CustomerID", "CountryCode"]].dropna(axis=0, how='any')

''' Data set of how much of each item each customers purchased. '''
reddataset = deepcopy(dataset[["StockCode", "giftBag", "InvoiceNo", "CustomerID", "InvoiceMonth", "CountryCode"]])

rawdataset = deepcopy(dataset[["StockCode", "giftBag", "InvoiceNo", "Quantity", "InvoiceMonth", "CountryCode"]])

'''Add a column to hold a value to count for being purchased'''
reddataset = reddataset.reindex(columns=np.append(reddataset.columns.values, "Count"))

'''If there is a line in data set, the item was purchased. So set all columns to 1.'''
reddataset["Count"] = 1
reddataset = reddataset[reddataset.StockCode != 23437]
reddataset = reddataset[reddataset.StockCode != 23375]


'''Get list of each customer, with the items and how many times those items were purchased.'''
reddataset = reddataset.groupby(["giftBag", "StockCode"]).count().reset_index()

reddataset = reddataset.sort_values(by=['StockCode', 'giftBag'])

'''
The below prints the customer purchasing frequency information by stock iteme
Inside the brackets customize what rows are printed. [X:Y], where X is start row
and Y is end row
'''
#reddataset = reddataset.drop(["giftBag"], axis=1)

#reddataset = reddataset[["StockCode", "Count"]]
reddataset = reddataset.reindex(columns=np.append(reddataset.columns.values, "YesBag"))
reddataset = reddataset.reindex(columns=np.append(reddataset.columns.values, "NoBag"))
reddataset = reddataset.reindex(columns=np.append(reddataset.columns.values, "BagPurchProb"))

reddataset["YesBag"] = 0
reddataset["NoBag"] = 0
reddataset["BagPurchProb"] = float(0)

for i in reddataset.index:
    if(reddataset.at[i, "giftBag"] == 1):
        reddataset.at[i, "YesBag"] = reddataset.at[i, "Count"]
    if(reddataset.at[i, "giftBag"] == 0):
            reddataset.at[i, "NoBag"] = reddataset.at[i, "Count"]

reddataset = reddataset.groupby(["StockCode"]).sum().reset_index()
reddataset = reddataset[["StockCode", "YesBag", "NoBag", "BagPurchProb"]]

for i in reddataset.index:
    x = reddataset.at[i, "YesBag"]
    y = reddataset.at[i, "NoBag"]
    z = x + y
    reddataset.at[i, "BagPurchProb"] = float(x)/float(z)

reddataset = reddataset.sort_values(["BagPurchProb", "YesBag"], ascending = False)
#print(reddataset.iloc[0:10])
test = reddataset.loc[reddataset["StockCode"] == 15039, ["BagPurchProb"]].values


'''Reduced data set size for faster testing, will expand to full data set eventually'''
dataset = dataset[0:]

'''Splits data into training and test sets'''
trainset, testset = train_test_split(rawdataset, test_size=0.9, random_state=12)
#trainset, testset = train_test_split(reddataset, test_size=0.5, random_state=int(time.time()))

'''Set up bayes classifier'''
gnb = GaussianNB()
#used_features = ["giftBag", "StockCode", "Quantity", "UnitPrice", "CustomerID"]
used_features = ["Quantity", "StockCode", "InvoiceMonth", "CountryCode"]
gnb.fit(trainset[used_features].values, trainset["giftBag"])

y_pred = gnb.predict(testset[used_features])

print(metrics.accuracy_score(testset["giftBag"], y_pred))


'''
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%".format(
          X_test.shape[0], (X_test["giftBag"] != y_pred).sum(),
          100*(1-(X_test["giftBag"] != y_pred).sum()/X_test.shape[0])
))
'''

mean_christmas_bag = np.mean(trainset["giftBag"])

mean_not_christmas_bag = 1 - mean_christmas_bag
'''
print("\nChristmas Bag purchased prob = {:03.2f}%, Not purchased prob = {:03.2f}%"
      .format(100*mean_christmas_bag, 100*mean_not_christmas_bag))
'''