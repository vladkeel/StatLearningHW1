import os
import numpy as np
from sklearn import datasets
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

iris = datasets.load_iris()
X = iris.data
y = iris.target
#Setosa:
y_setosa = []
for i in range(y.size):
    if y[i] == 0:
        y_setosa.append(1)
    else:
        y_setosa.append(-1)
y_setosa = np.array(y_setosa)
X_train_s, X_test_s, y_train_s, y_test_s = model_selection.train_test_split(X, y_setosa, random_state=1000)
setosa_clf = LogisticRegression().fit(X_train_s, y_train_s)
print("Results for setosa classifier {}".format(setosa_clf.score(X_test_s, y_test_s)))
y_versicolour = []
for i in range(y.size):
    if y[i] == 1:
        y_versicolour.append(1)
    else:
        y_versicolour.append(-1)
y_versicolour = np.array(y_versicolour)
X_train_ve, X_test_ve, y_train_ve, y_test_ve = model_selection.train_test_split(X, y_versicolour, random_state=1000)
versicolour_clf = LogisticRegression().fit(X_train_ve, y_train_ve)
print("Results for versicolour classifier {}".format(versicolour_clf.score(X_test_ve, y_test_ve)))
y_virginicacv = []
for i in range(y.size):
    if y[i] == 2:
        y_virginicacv.append(1)
    else:
        y_virginicacv.append(-1)
y_virginicacv = np.array(y_virginicacv)
X_train_vi, X_test_vi, y_train_vi, y_test_vi = model_selection.train_test_split(X, y_virginicacv, random_state=1000)
virginicacv_clf = LogisticRegression().fit(X_train_vi, y_train_vi)
print("Results for virginicacv classifier {}".format(virginicacv_clf.score(X_test_vi, y_test_vi)))

def one_vs_rest(clf_s, clf_ve, clf_vi, X):
    results = []
    prob_s = clf_s.predict_proba(X)
    prob_ve = clf_ve.predict_proba(X)
    prob_vi = clf_vi.predict_proba(X)
    print("129 is: {}.".format(X[129]))
    print("prob for ve is {}".format(prob_ve[129][1]))
    print("prob from vi is {}".format(prob_vi[129][1]))
    for i in range(np.size(X,0)):
        if prob_s[i][1] >= prob_ve[i][1] and prob_s[i][1] >= prob_vi[i][1]:
            results.append(0)
        elif prob_ve[i][1] >= prob_s[i][1] and prob_ve[i][1] >= prob_vi[i][1]:
            results.append(1)
        elif prob_vi[i][1] >= prob_s[i][1] and prob_vi[i][1] >= prob_ve[i][1]:
            results.append(2)
    return results

y_pred = one_vs_rest(setosa_clf, versicolour_clf, virginicacv_clf, X)
for i in range(len(y_pred)):
    print("{}: {}".format(i,y_pred[i]))
print(confusion_matrix(y, y_pred))
