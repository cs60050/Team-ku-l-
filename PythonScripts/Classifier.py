### This script uses various classifiers and compares their accuracy and run time

import itertools
import numpy as np
import pandas as pd
from time import time
from sklearn import metrics
import matplotlib.pyplot as plt
from numpy.random import random

from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Perceptron
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier


from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier


from sklearn.feature_selection import SelectKBest, chi2

### Import the Features Vector
df = pd.read_csv("features3.csv",delimiter="\t")

### Keep only newspapers you want to compare. Drop Others.
'''
df=df[df.newspaper!="ET"]
df=df[df.newspaper!="IEx"]
df=df[df.newspaper!="FEx"]
df=df[df.newspaper!="DC"]
'''
##print(df.groupby('newspaper').count())

Y = df["newspaper"].values  ## Extracting Newspaper Names which is our class

## First 3 columns aren't features.
##They are index, id and newspaper names.
cols = [0,1,2]

X = df.drop(df.columns[cols], axis=1)   ## So, creating a dataframe with only the features
X = X.as_matrix()                       ## Converting dataframeto a matrix to make compatible with all functions


### Splitting Data into Train and Test using a StratifiedShuffleSplit
sss = StratifiedShuffleSplit(Y, 10, test_size=0.3, random_state=0)

### Using the generated indices to create Test and Train datasets
for train_index, test_index in sss:
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]


### Select K best features based on a chi-squared test
feature_chi = 72    ## selecting best 2/3rd features
chi2 = SelectKBest(chi2, k=feature_chi)
X_train = chi2.fit_transform(X_train, y_train)
X_test  = chi2.transform(X_test)
#print(X_train)


### Defining a function to print statistics which helps us benchmark classifier performance
def benchmark(clf):
    clf_descr = str(clf).split('(')[0]  ## store name of the classifier
    print(clf_descr)                    ## print name
    t0 = time()                     ## store current time in t0
    clf.fit(X_train, y_train)       ## run classifier to fit data
    train_time = time() - t0        ## Calculate time take to train
    print("train time: %0.3fs" % train_time)    ## print statistic

    t0 = time()                     ## store current time in t0
    pred = clf.predict(X_test)      ## use trained classifer to predict class for test data
    test_time = time() - t0         ## Calculate time take to predict for test data
    print("test time:  %0.3fs" % test_time)     ## print statistic

    score = metrics.accuracy_score(y_test, pred)    ## calculate accuracy
    print("accuracy:   %0.3f" % score)              ## print accuracy

    #clf_descr = str(clf).split('(')[0]      ## store name of the classifier


    ## Plot Confusion Matrix
    newsagencies=["Guardian", "TOI", "FEx", "IEx", "DC", "ET"]
    plt.figure()
    confusionMatrix = confusion_matrix(y_test, pred, labels=newsagencies)
    plt.imshow(confusionMatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(clf_descr)
    plt.colorbar()
    tick_marks = np.arange(len(newsagencies))
    plt.xticks(tick_marks, newsagencies, rotation=45)
    plt.yticks(tick_marks, newsagencies)
    print(confusionMatrix)
    thresh = confusionMatrix.max() / 2
    for i, j in itertools.product(range(confusionMatrix.shape[0]), range(confusionMatrix.shape[1])):
        plt.text(j, i, confusionMatrix[i, j],
                 horizontalalignment="center",
                 color="white" if confusionMatrix[i, j] > thresh else "black")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    ## plotting of confusion matrix complete here

    print(" ")
    return clf_descr, score, train_time, test_time
### Function Definition complete
#
#
### Declaring an array to store all results
results = []
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (RandomForestClassifier(n_estimators=100), "Random forest"),
        (ExtraTreesClassifier(n_estimators=10), "Extra Trees"),
        (AdaBoostClassifier(n_estimators=100),"Ada Boost")):
    results.append(benchmark(clf))

results.append(benchmark(BernoulliNB(alpha=.01)))   ## Since BernoulliNB is not iterbale we do it separately
results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50, penalty="elasticnet")))

plt.show()

#print(results)
