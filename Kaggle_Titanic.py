# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:24:06 2017

@author: koisj_000
"""

# Importing packages

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# Acquire data
train_df = pd.read_csv('C:/Users/koisj_000/Desktop/Programowanie/Python/Kaggle/Titanic/train.csv')
test_df = pd.read_csv('C:/Users/koisj_000/Desktop/Programowanie/Python/Kaggle/Titanic/test.csv')
combine = [train_df, test_df]

# Column names
train_df.columns

# preview the data
train_df.head()
train_df.tail()

# Data types
train_df.info()
print('_'*40)
test_df.info()

# Distribution of numerical features
train_df.describe()

# Distribution of categorical features
train_df.describe(include=['O'])

# Wrangle data

# Dropping Ticket and Cabin
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# Extracting Title
for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

# More common titles
for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

# Converting caterogical to ordinal
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df.head()

# Dropping Name feature and PassangerId from dataset
train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

# Converting caterogical feature
for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

train_df.head()

# Completing Age feature using Pclass and Sex feature
guess_ages = np.zeros((2,3))
guess_ages

for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5
            
    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)

train_df.head()

# Creating AgeBand
train_df['AgeBand'] = pd.cut(train_df['Age'], 5)
train_df[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

# Creating Age bands
for dataset in combine:    
    dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[ dataset['Age'] > 64, 'Age']
train_df.head()

# removeing AgeBand feature from dataset
train_df = train_df.drop(['AgeBand'], axis=1)
combine = [train_df, test_df]
train_df.head()

# Creating feature FamilySize
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Creating feature IsAlone
for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

train_df[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()

# Dropping Parch, SibSp and FamilySize in favour of IsAlone
train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

train_df.head()

# Creatinf artificial feature
for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass

train_df.loc[:, ['Age*Class', 'Age', 'Pclass']].head(10)

# Completing Embarked feature
freq_port = train_df.Embarked.dropna().mode()[0]
freq_port

for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Converting categorical feature to numeric
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

train_df.head()

# Completing Fare feature
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
test_df.head()

# Creating FareBand
train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

# Convert the Fare feature to ordinal values based on the FareBand
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
combine = [train_df, test_df]
    
train_df.head(10)
test_df.head(10)

# Preparing data for modelling
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test  = test_df.drop("PassengerId", axis=1).copy()
X_train.shape, Y_train.shape, X_test.shape


############################## Random Forest #############################
random_forest = RandomForestClassifier(n_estimators=1000, max_features= 0.8, min_samples_leaf=15, oob_score=True, n_jobs = -1)
random_forest.fit(X_train, Y_train)
Y_pred = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
random_forest.oob_score_ # likely score on test set
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
acc_random_forest

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = random_forest, X = X_train, y = Y_train, cv = 10)
accuracies.mean()

## Searching for the best parameters with RandomizedSearchCV
#from sklearn.model_selection import RandomizedSearchCV
#
## specify "parameter distributions"
#max_depth = list(range(4, 9))
#min_samples_leaf= list(range(5, 11))
#param_dist = dict(min_samples_leaf = min_samples_leaf)
#
#random_forest = RandomForestClassifier(n_estimators=1000, n_jobs = -1)
#rand = RandomizedSearchCV(random_forest, param_dist, cv=10, scoring='accuracy', n_iter=5, return_train_score=False)
#rand.fit(X_train, Y_train)
#pd.DataFrame(rand.cv_results_)[['mean_test_score', 'std_test_score', 'params']]

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('C:/Users/koisj_000/Desktop/Programowanie/Python/Kaggle/Titanic/submission_random_forest_tunning_v6.csv', index=False)

############################## XGBoost ##############################
# from: https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/

from xgboost.sklearn import XGBClassifier
xgboost = XGBClassifier(
 #learning_rate =0.01,
 n_estimators=2000,
 max_depth=4, # 4-6 good for starting points
 min_child_weight=2, # higher values to prevent overfiting
 gamma=0.9, # 0-0.2 for starting point
 subsample=0.8, # commonly used for starting point
 colsample_bytree=0.8, # commonly used for starting point
 objective= 'binary:logistic',
 scale_pos_weight=1, # A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence
 seed=0)

xgboost.fit(X_train, Y_train)
Y_pred = xgboost.predict(X_test)
xgboost.score(X_train, Y_train)
acc_xgboost = round(xgboost.score(X_train, Y_train) * 100, 2)
acc_xgboost

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = xgboost, X = X_train, y = Y_train, cv = 10)
accuracies.mean()


submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('C:/Users/koisj_000/Desktop/Programowanie/Python/Kaggle/Titanic/submission_xgboost_v2.csv', index=False)


############################### Soft voting classifier ##############################
#
#from sklearn.ensemble import VotingClassifier
#
#voting_clf = VotingClassifier(estimators = [('rf', random_forest), ('xgb_clf', xgboost)], voting = 'soft')
#voting_clf.fit(X_train, Y_train)
#Y_pred = voting_clf.predict(X_test)
#voting_clf.score(X_train, Y_train)
#
## Applying k-Fold Cross Validation
#from sklearn.model_selection import cross_val_score
#accuracies = cross_val_score(estimator = voting_clf, X = X_train, y = Y_train, cv = 10)
#accuracies.mean()
#
#submission = pd.DataFrame({
#        "PassengerId": test_df["PassengerId"],
#        "Survived": Y_pred
#    })
#
#submission.to_csv('C:/Users/koisj_000/Desktop/Programowanie/Python/Kaggle/Titanic/submission_soft_voting_classifier.csv', index=False)