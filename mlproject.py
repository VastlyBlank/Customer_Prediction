import pandas as pd
from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Settings to be able to print all the columns of data set
pd.set_option("display.max_columns", 15)
pd.set_option("display.width", 200)
pd.options.display.max_rows = 400

# Import data file. testdata.csv was scrubbed to remove StockCodes with letters,
# negative quantities, and blank descriptions.
df = pd.read_csv('testdata2.csv', encoding="ISO-8859-1", dtype={'CustomerID': str, 'InvoiceID': str})
dataset = df[["giftBag", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceMonth",
              "UnitPrice", "CustomerID", "CountryCode"]].dropna(axis=0, how='any')

# Copy of data set containing only relevant columns
rawdataset = deepcopy(dataset[["StockCode", "giftBag", "InvoiceNo", "Quantity", "InvoiceMonth", "CountryCode"]])

# Splits data into training and test sets
trainset, testset = train_test_split(rawdataset, test_size=0.2, random_state=12)

# Features used for our tests
used_features = ["StockCode", "Quantity", "InvoiceMonth", "CountryCode"]
used_features2 = ["StockCode", "Quantity", "InvoiceMonth"]

'''Gaussian Bayes classifier'''
# GNB for giftBag predictions
gnb = MultinomialNB()

# GNB for CountryCode predictions
gnb2 = MultinomialNB()

# Fit both sets
gnb.fit(trainset[used_features].values, trainset["giftBag"])
gnb2.fit(trainset[used_features2].values, trainset["CountryCode"])

# giftBag predictions
y_pred = gnb.predict(testset[used_features])

# CountryCode predictions
y_pred2 = gnb2.predict(testset[used_features2])

# Accuracy of giftBag predictions
yacc = metrics.accuracy_score(testset["giftBag"], y_pred)

# Accuracy of CountryCode predictions
yacc2 = metrics.accuracy_score(testset["CountryCode"], y_pred2)

'''Decision Tree'''
# DT for giftBag predictions
tree_stuff = tree.DecisionTreeClassifier()

# DT for CountryCode predictions
tree_stuff2 = tree.DecisionTreeClassifier()

# Fit both trees
tree_stuff.fit(trainset[used_features].values, trainset["giftBag"])
tree_stuff2.fit(trainset[used_features2].values, trainset["CountryCode"])

# giftBag predictions
t_pred = tree_stuff.predict(testset[used_features])

# CountryCode predictions
t_pred2 = tree_stuff2.predict(testset[used_features2])

# Accuracy of giftBag predictions
tacc = metrics.accuracy_score(testset["giftBag"], t_pred)

# Accuracy of CountryCode predictions
tacc2 = metrics.accuracy_score(testset["CountryCode"], t_pred2)

'''K-NN'''
# KNN for giftBag predictions
knn = KNeighborsClassifier(n_neighbors=15)

# KNN for CountryCode predictions
knn2 = KNeighborsClassifier(n_neighbors=15)

# Fit both sets
knn.fit(trainset[used_features].values, trainset["giftBag"])
knn2.fit(trainset[used_features2].values, trainset["CountryCode"])

# giftBag predictions
k_pred = knn.predict(testset[used_features])

# CountryCode predictions
k_pred2 = knn2.predict(testset[used_features2])

# Accuracy of giftBag predictions
knnacc = metrics.accuracy_score(testset["giftBag"], k_pred)

# Accuracy of CountryCode predictions
knnacc2 = metrics.accuracy_score(testset["CountryCode"], k_pred2)

'''Printing Results'''
print("\nGift Bag Accuracy Results:")
print("GNB Accuracy: {0:.2f}%".format(yacc*100))
print("Decision Tree Accuracy: {0:.2f}%".format(tacc*100))
print("KNN Accuracy: {0:.2f}%".format(knnacc*100))

print("\nCountryCode Accuracy Results:")
print("GNB Accuracy: {0:.2f}%".format(yacc2*100))
print("Decision Tree Accuracy: {0:.2f}%".format(tacc2*100))
print("KNN Accuracy: {0:.2f}%".format(knnacc2*100))
