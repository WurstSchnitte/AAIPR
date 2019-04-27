import pandas as pd
from sklearn.linear_model import LogisticRegression
from preprocessing import preprocess
from postprocessing import postprocess
# split function to split the data into training and test data
from sklearn.model_selection import train_test_split
# function to get the accuracy for the trained model
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
import numpy as np

def accTestLinearReg(all_X, all_y):
    train_X, test_X, train_y, test_y = train_test_split(
        all_X, all_y, test_size=0.2, random_state=0)

    # object for logic regression
    lr = LogisticRegression()

    lr.fit(train_X, train_y)
    predictions = lr.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)

    scores = cross_val_score(lr, all_X, all_y, cv=10)
    np.mean(scores)

    print('Lr accuracy: ', accuracy)
    print('Lr mean :', np.mean(scores))
    return

def accTestTreePredict(all_X, all_y):
    train_X, test_X, train_y, test_y = train_test_split(
        all_X, all_y, test_size=0.2, random_state=0)

    # object for logic regression
   
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_X, train_y)
    predictions = clf.predict(test_X)
    accuracy = accuracy_score(test_y, predictions)

    scores = cross_val_score(clf, all_X, all_y, cv=10)
    np.mean(scores)

    print('Tree accuracy: ', accuracy)
    print('Tree mean :', np.mean(scores))
    return

def linearReg(data, all_1, all_2, columns):
    lr = LogisticRegression()
    lr.fit(all_1, all_2)
    holdout_predictions = lr.predict(data[columns])
    return holdout_predictions

def treePredict(data, all_1, all_2, columns):
    clf = tree.DecisionTreeClassifier()
    clf.fit(all_1, all_2)
    holdout_predictions = clf.predict(data[columns])
    return holdout_predictions

# load the data
test_dataSet = pd.read_csv("test.csv")
train_dataSet = pd.read_csv("train.csv")

cleanup_test = preprocess(test_dataSet)
test_dataSet = cleanup_test[0]

cleanup_train = preprocess(train_dataSet)
train_dataSet = cleanup_train[0]

# learn out model with columns
columns = cleanup_train[1]

all_X = train_dataSet[columns]
all_y = train_dataSet['Survived']

holdout = test_dataSet

linearReg(holdout, all_X, all_y, columns)
holdout_predictions = treePredict(holdout, all_X, all_y, columns)

submission = postprocess(holdout, holdout_predictions)

submission.to_csv('titanic_submission.csv', index=False)

accTestLinearReg(all_X,all_y)
accTestTreePredict(all_X, all_y)