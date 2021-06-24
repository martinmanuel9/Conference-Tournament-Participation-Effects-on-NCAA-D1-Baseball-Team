#SVM
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import svm

dataset = np.loadtxt('Delta_RPI_Analysis_no_titles.csv', delimiter=',')
X = dataset[:, :-1]
y = dataset[:, -1]

bucket1 = []
for i in range(X):
    bucket1.append(0)

bucket2 = []
for i in range(X):
    bucket2.append(1)

X = np.vstack([X,y])
y = np.hstack([bucket1, bucket2])

data_train, data_test, labels_train, labels_test = train_test_split(X, y)


clf = svm.SVC(kernel='linear')
clf.fit(data_train, labels_train)
predictions = []


for i in range(data_test.shape[0]):
    predictions.append(clf.predict([[data_test[i,0], data_test[i,1]]]))

training_plot = plt.subplot()
training_plot.scatter(data_train[:,0], data_train[:,1], marker=".")

for i in range(data_train.shape[0]):
    if labels_train[i] == 0:
        training_plot.scatter(data_train[i,0], data_train[i,1], marker=".", color=None)
    else:
        training_plot.scatter(data_train[i,0], data_train[i,1], marker="o", color=None)
plt.title('SVM: Linear')
plt.show()

error = 0

for i in range(data_test.shape[0]):
    if clf.predict([[data_test[i,0], data_test[i,1]]]) != labels_test[i]:
        error += 1

print("Classifier Error Linear SVM: " + str(error))

clf2 = svm.SVC(kernel='rbf')
clf2.fit(data_train, labels_train)
predictions2 = []


for i in range(data_test.shape[0]):
    predictions2.append(clf2.predict([[data_test[i,0], data_test[i,1]]]))

training_plot = plt.subplot()
training_plot.scatter(data_train[:,0], data_train[:,1], marker=".")

for i in range(data_train.shape[0]):
    if labels_train[i] == 0:
        training_plot.scatter(data_train[i,0], data_train[i,1], marker=".", color=None)
    else:
        training_plot.scatter(data_train[i,0], data_train[i,1], marker="o", color=None)
plt.title('SVM: RBF')
plt.show()

error2 = 0

for i in range(data_test.shape[0]):
    if clf2.predict([[data_test[i,0], data_test[i,1]]]) != labels_test[i]:
        error2 += 1

print("Classifier Error RBF: " + str(error2))