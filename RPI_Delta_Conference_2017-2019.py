#
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn import linear_model
from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm


# load dataset
dataset = np.loadtxt('Delta_RPI_Analysis_no_titles.csv', delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1]

# preprocess data
x = MinMaxScaler().fit_transform(X)

clf1 = linear_model.LinearRegression()
clf2= linear_model.LogisticRegression()
scalar = StandardScaler()
adaClf = AdaBoostClassifier(n_estimators=100, random_state=0)
adaClf.fit(X, y)

cv = ShuffleSplit(n_splits=10)
scalar = StandardScaler()
scalar.fit(X)
scaled_data = scalar.transform(X)
pca = PCA()
pca.fit(scaled_data)
x_pca = pca.transform(scaled_data)
pca_score = pca.score(x_pca, scaled_data)
scores1 = cross_val_score(clf1, x, y, cv=cv, verbose=5)
scores2 = cross_val_score(clf2, x,y, cv=cv, verbose=5)

print("Linear Regression (Accuracy) Score: ", np.mean(scores1))
print("Linear Regression Average Error: ", (1-np.mean(scores1)))
print("Logistic Regression Accuracy Score: ", np.mean(scores2))
print("Logistic Regression Average Error: ", (1-np.mean(scores2)))
#print('PCA Accuracy: ', pca_score)
#print("PCA Average Error", (1-pca_score))
print("Adaboost Accuracy Score", adaClf.score(X, y))
print("Adaboost Average Error:,", (1-adaClf.score(X,y)))