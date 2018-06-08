import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

'''Settings to be able to print all the columns of data set'''
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)
pd.options.display.max_rows = 400

'''
    Import data file. testdata.csv was scrubbed to remove StockCodes with letters, 
    negative quantities, and blank descriptions.
'''
df = pd.read_csv('testdata2.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
dataset = df[["giftBag", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceMonth",
              "UnitPrice", "CustomerID", "CountryCode"]].dropna(axis=0, how='any')

# Copy of data set containing only relevant columns
rawdataset = deepcopy(dataset[["StockCode", "giftBag", "InvoiceNo", "Quantity", "InvoiceMonth", "CountryCode"]])

# Splits data into training and test sets
trainset, testset = train_test_split(rawdataset, test_size=0.2, random_state=12)

# Features used for our tests
used_features = ["StockCode"]
used_features2 = ["StockCode", "Quantity", "InvoiceMonth", "CountryCode"]

# Stock code used for CountryCode predictions
stCode = 22902

'''Gaussian Bayes classifier'''
# GNB for CountryCode Prediction
gnb = MultinomialNB()

# GNB for giftBag predictions
gnb2 = MultinomialNB()

# Fit both sets
gnb.fit(trainset[used_features].values, trainset["CountryCode"])
gnb2.fit(trainset[used_features2].values, trainset["giftBag"])

# Predict Country Code
y_pred = gnb.predict(stCode)

# giftBag predictions
y_pred2 = gnb2.predict(testset[used_features2])

# Accuracy of giftBag Predictions
yacc = metrics.accuracy_score(testset["giftBag"], y_pred2)

'''Decision Tree'''
# DT for CountryCode predictions
tree_stuff = tree.DecisionTreeClassifier()

# DT for giftBag predictions
tree_stuff2 = tree.DecisionTreeClassifier()

# Fit both trees
tree_stuff.fit(trainset[used_features].values, trainset["CountryCode"])
tree_stuff2.fit(trainset[used_features2].values, trainset["giftBag"])

# Predict Country Code
t_pred = tree_stuff.predict(stCode)

# giftBag predictions
t_pred2 = tree_stuff2.predict(testset[used_features2])

# Accuracy of giftBag Predictions
tacc = metrics.accuracy_score(testset["giftBag"], t_pred2)

'''K-NN'''
# KNN for CountryCode Prediction
knn = KNeighborsClassifier(n_neighbors=1)

# KNN for giftBag predictions
knn2 = KNeighborsClassifier(n_neighbors=1)

# Fit both sets
knn.fit(trainset[used_features].values, trainset["CountryCode"])
knn2.fit(trainset[used_features2].values, trainset["giftBag"])

# Predict Country Code
k_pred = knn.predict(stCode)

# giftBag predictions
k_pred2 = knn2.predict(testset[used_features2])

# Accuracy of giftBag Predictions
knnacc = metrics.accuracy_score(testset["giftBag"], k_pred2)

'''Printing Results'''
print("CountryCode Predictions:")
print("GNB: ", y_pred)
print("Decision Tree: ", t_pred)
print("KNN: ", k_pred)

print("\nAccuracy Results:")
print("GNB giftBag Accuracy: ", int(yacc*100), "%")
print("Decision Tree giftBag Accuracy: ", int(tacc*100), "%")
print("KNN giftBag Accuracy: ", int(knnacc*100), "%")
