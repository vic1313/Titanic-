# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 01:24:31 2018

@author: vibhanshu vaibhav
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.metrics import confusion_matrix


df_train=pd.read_csv('train.csv',sep=',')
df_test=pd.read_csv('test.csv',sep=',')

df_train.fillna(0, inplace=True)
df_test.fillna(0,inplace=True)


df_train.dropna()

df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['Age', 'Survived']].groupby(['Age'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['SibSp', 'Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['Parch', 'Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_train[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False)

gender={'male':0,'female':1}
Embark={'C':1,'Q':2,'S':3,0:0}
df_train.Sex= [gender[item] for item in df_train.Sex]
df_train.Embarked=[Embark[item] for item in df_train.Embarked]
df_test.Sex= [gender[item] for item in df_test.Sex]
df_test.Embarked=[Embark[item] for item in df_test.Embarked]


df_train['newCabin'] = np.where(df_train['Cabin'] != 0, 1, df_train['Cabin'])
df_test['newCabin'] = np.where(df_test['Cabin'] != 0, 1, df_test['Cabin'])

from sklearn.linear_model import LogisticRegression
X=df_train.drop(['Name','Survived','Ticket','Fare','Cabin'],axis=1)
Y=df_train['Survived']
x=df_test.drop(['Name','Ticket','Fare','Cabin'],axis=1)
pca = PCA(n_components=2).fit_transform(X)

model = LogisticRegression()
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=20)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=5000)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
zz=model.predict(X_test)
confusion = confusion_matrix(Y_test, zz)
print(confusion)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)
zz=model.predict(x)

from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()
model.fit(X,Y)
model.score(X,Y)
zz=model.predict(x)


from sklearn.svm import SVC

model = SVC(kernel='linear')
model.fit(X,Y)
model.score(X,Y)
ss=model.predict(x)

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=1)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.svm import SVC

model = SVC(kernel='rbf', C=10)
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train,Y_train)
model.score(X_test,Y_test)

from sklearn.neural_network import MLPClassifier

model = MLPClassifier()
model.fit(X,Y)
model.score(X,Y)
cc=model.predict(x)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X,Y)
zz=model.predict(x)

col=['PassengerId','Survived']
df_csv = pd.DataFrame(columns=col)
df_csv.PassengerId=df_test.PassengerId
df_csv.Survived=ss

df_csv.to_csv('Titanic_survivor.csv',index=False)
