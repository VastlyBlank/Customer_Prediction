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
used_features = ["StockCode", "Quantity", "InvoiceMonth", "CountryCode"]
used_features2 = ["StockCode"]

# Stock code used for CountryCode predictions
stCode = 22802

'''Gaussian Bayes classifier'''
# GNB for giftBag predictions
gnb = MultinomialNB()

# GNB for CountryCode Prediction
gnb2 = MultinomialNB()

# Fit both sets
gnb.fit(trainset[used_features].values, trainset["giftBag"])
gnb2.fit(trainset[used_features2].values, trainset["CountryCode"])

# giftBag predictions
y_pred = gnb.predict(testset[used_features])

# Predict Country Code
y_pred2 = gnb2.predict(stCode)
# Used for Accuracy
y_pred3 = gnb2.predict(testset[used_features2])

# Accuracy of Predictions
yacc = metrics.accuracy_score(testset["giftBag"], y_pred)
yacc2 = metrics.accuracy_score(testset["CountryCode"], y_pred3)

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

# Predict Country Code
t_pred2 = tree_stuff2.predict(stCode)
# Used for CountryCode Accuracy
t_pred3 = tree_stuff2.predict(testset[used_features2])

# Accuracy of Predictions
tacc = metrics.accuracy_score(testset["giftBag"], t_pred)
tacc2 = metrics.accuracy_score(testset["CountryCode"], t_pred3)

'''K-NN'''
# KNN for giftBag predictions
knn = KNeighborsClassifier(n_neighbors=1)

# KNN for CountryCode Prediction
knn2 = KNeighborsClassifier(n_neighbors=1)

# Fit both sets
knn.fit(trainset[used_features].values, trainset["giftBag"])
knn2.fit(trainset[used_features2].values, trainset["CountryCode"])


# giftBag predictions
k_pred = knn.predict(testset[used_features])
# Predict Country Code
k_pred2 = knn2.predict(stCode)
# Used for CountryCode Accuracy
k_pred3 = knn2.predict(testset[used_features2])

# Accuracy of giftBag Predictions
knnacc = metrics.accuracy_score(testset["giftBag"], k_pred)
knnacc2 = metrics.accuracy_score(testset["CountryCode"], k_pred3)


'''Printing Results'''
print("\nGift Bag Accuracy Results:")
print("GNB Accuracy: {0:.2f}%".format(yacc*100))
print("Decision Tree Accuracy: {0:.2f}%".format(tacc*100))
print("KNN Accuracy: {0:.2f}%".format(knnacc*100))

print("\nCountryCode Predictions:")
print("GNB: ", y_pred2)
print("Decision Tree: ", t_pred2)
print("KNN: ", k_pred2)

print("\nCountryCode Accuracy Results:")
print("GNB Accuracy: {0:.2f}%".format(yacc2*100))
print("Decision Tree Accuracy: {0:.2f}%".format(tacc2*100))
print("KNN Accuracy: {0:.2f}%".format(knnacc2*100))

