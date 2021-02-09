from sklearn.ensemble import RandomForestClassifier
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier


# X
training_data = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/Features_training_15122020.csv"))
# x_test
test_data = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/Features_validation_15122020.csv"))

# Y
training_labels = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/label_groundtruth_train.csv"))
# Y_test
test_labels = pd.read_csv(os.path.join(
    os.getcwd(), "csv_features/label_groundtruth_validation.csv"))

X_labels = ['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
            't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
            'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
            'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03']

Y_labels = ['label']

# Standardlize data

scaler = StandardScaler()
X_Train = scaler.fit(training_data[X_labels])
X_Train = scaler.transform(training_data[X_labels])

X_Train = pd.DataFrame(X_Train, columns=['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
                                         't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
                                         'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
                                         'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'])

X_Train = pd.concat([training_data['imageID'], X_Train], axis=1)


scaler = StandardScaler()
X_Test = scaler.fit(test_data[X_labels])
X_Test = scaler.transform(test_data[X_labels])

X_Test = pd.DataFrame(X_Test, columns=['mean-R', 'skew-R', 'std-R', 'entropy-R', 'mean-G', 'skew-G', 'std-G', 'entropy-G', 'mean-B', 'skew-B', 'std-B', 'entropy-B',
                                       't1', 't2', 't3', 't4', 't5', 't6', 't7', 't8', 't9', 't10', 't11', 't12', 't13', 't14', 't15', 't16',
                                       'hu1', 'hu2', 'hu3', 'hu4', 'hu5', 'hu6', 'hu7',
                                       'm00', 'm10', 'm01', 'm20', 'm11', 'm02', 'm30', 'm21', 'm12', 'm03'])

X_Test = pd.concat([test_data['imageID'], X_Test], axis=1)


Y_Train = training_labels[Y_labels]
Y_Test = test_labels[Y_labels]

# #Logistic Regression
# model = LogisticRegression(max_iter=800)

# model.fit(X_Train[X_labels],Y_Train.values.ravel())

# Y_Pred = model.predict(X_Test[X_labels])

# # count_misclassified = (Y_Test != Y_Pred).sum()
# # print('Misclassified samples: {}'.format(count_misclassified))

# accuracy = accuracy_score(Y_Test, Y_Pred)
# print('Accuracy: {:.2f}'.format(accuracy))

# print(classification_report(Y_Test, Y_Pred, labels =[1,2,3,4,5,6,7,8,9]))


# #Support Vector Machines
# classifier_svm = svm.SVC()
# classifier_svm.fit(X_Train[X_labels],Y_Train.values.ravel())

# Y_Pred_svm = classifier_svm.predict(X_Test[X_labels])

# print(classification_report(Y_Test, Y_Pred_svm, labels =[1,2,3,4,5,6,7,8,9]))


# KNearestNeighbours
neigh = KNeighborsClassifier(
    n_neighbors=17, weights='distance', algorithm='ball_tree', leaf_size=100)
Y_pred_KNN = neigh.fit(
    X_Train[X_labels], Y_Train.values.ravel()).predict((X_Test[X_labels]))

accuracy = accuracy_score(Y_Test, Y_pred_KNN)
print('Accuracy: {:.2f}'.format(accuracy))
print(classification_report(Y_Test, Y_pred_KNN,
                            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9], zero_division=0))
# Accuracy 60%


""" # Random Forest
clf = RandomForestClassifier(max_depth=7, random_state=0,
                             n_estimators=1000, max_features='sqrt', class_weight='balanced')
clf.fit(X_Train[X_labels], Y_Train.values.ravel())
Y_pred_RF = clf.predict((X_Test[X_labels]))

accuracy = accuracy_score(Y_Test, Y_pred_RF)
print('Accuracy: {:.2f}'.format(accuracy))

print(classification_report(Y_Test, Y_pred_RF,
                            labels=[1, 2, 3, 4, 5, 6, 7, 8, 9]))
 """
