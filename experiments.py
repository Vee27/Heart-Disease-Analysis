"""
            Heart Disease Analysis using ML algorithms from scikit-learn
    (Logistic Regression, SVM, KNN, Gaussian Naiive Bayes, Decision Trees, Random Forest)
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier


# Read data
df = pd.read_csv(r'data/heart.csv')
# print(df.head())

# first step is to convert all categorical to numerical values
a = pd.get_dummies(df['cp'], prefix="cp")  # representing chest pain [0-3]
b = pd.get_dummies(df['thal'], prefix="thal")  # thalassemia [0-2]
c = pd.get_dummies(df['slope'], prefix="slope")  # [0-2]

# append the converted numerical values as new columns
temp_df = [df, a, b, c]
df = pd.concat(temp_df, axis=1)

# drop the columns with the categorical values
df.drop(columns=['cp', 'thal', 'slope'])

# dropping the class / target labels from the train data
X = df.drop(['target'], axis=1)

y = df.target.values

# Normalize
x = (X - np.min(X)) / (np.max(X) - np.min(X)).values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)


def train_and_predict(classifier):
    # fit the train data to classifier
    model = classifier.fit(x_train, y_train)

    # predict on test data
    predictions = model.predict(x_test)

    # evaluate the model
    score_ = model.score(x_test, y_test)
    conf = confusion_matrix(y_test, predictions)
    report = classification_report(y_test, predictions)

    print(conf)
    print(report)
    return score_ * 100


# Accuracy with Logistic Regression Classifier
acc_lr = train_and_predict(LogisticRegression())
print("Test Accuracy {:.2f}%".format(acc_lr))


# KNN Model
acc_knn = train_and_predict(KNeighborsClassifier(n_neighbors=2))  # n_neighbors means the k value
print("{} NN Test Accuracy: {:.2f}%".format(2, acc_knn))

# to find best k value
scoreList = []
for i in range(1, 20):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)

    # append all the scores
    scoreList.append(knn.score(x_test, y_test))

plt.plot(range(1, 20), scoreList)
plt.xticks(np.arange(1, 20, 1))
plt.xlabel("K value")
plt.ylabel("Score")
plt.title('Score with different K Values')
plt.show()

# print the maximum score
acc = max(scoreList) * 100
print("Maximum KNN Score is {:.2f}%".format(acc))

# SVM
acc_svm = train_and_predict(SVC(random_state = 1))
print("Test Accuracy with SVM Algorithm: {:.2f}%".format(acc_svm))

# Naive-Bayes
acc_nb = train_and_predict(GaussianNB())
print("Test Accuracy with Naive Bayes: {:.2f}%".format(acc))

# Decision Tree Classifier
acc_dt = train_and_predict(DecisionTreeClassifier())
print("Test Accuracy with Decision Tree: {:.2f}%".format(acc_dt))

# Random Forest Classification
acc_rf = train_and_predict(RandomForestClassifier(n_estimators=100, random_state=1))
print("Test Accuracy with Random Forest: {:.2f}%".format(acc_rf))

# plot the accuracies of all the classifiers

# add all the classifiers a s a dictionary
accuracies = {'Logistic Regression': acc_lr, 'KNN': acc_knn, 'SVM': acc_svm, 'Gaussian NB': acc_nb,
              'Decision Tree': acc_dt, 'Random Forest': acc_rf}
colors = ["blue", "green", "orange", "yellow", "lightblue", "lightgreen"]
plt.figure(figsize=(12, 4))
plt.yticks(np.arange(0, 100, 10))
plt.ylabel("Accuracy (in %)")
plt.xlabel("Classifiers")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors)
plt.show()
