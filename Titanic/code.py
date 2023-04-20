import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def replace_Age(row):
    if np.isnan(row['Age']):
        if row['Pclass']==1:
            row['Age'] = m1
        elif row['Pclass']==2:
            row['Age'] = m2
        else:
            row['Age'] = m3
    return row

def convert_sex(x):
    if x=='male':
        return 1
    return 0

def NullEmbarked(x):
    if x is None:
        return 'S'
    return x

def clear(df):
    temp1 = df[df.Pclass==1]
    temp2 = df[df.Pclass==2]
    temp3 = df[df.Pclass==3]
    m1 = np.mean(temp1.Age)
    m2 = np.mean(temp2.Age)
    m3 = np.mean(temp3.Age)
    df1 = df.apply(replace_Age, axis=1)
    df2 = df1.drop(['Name', 'Ticket', 'Cabin'], axis=1)
    df2['male'] = df2['Sex'].apply(convert_sex)
    df2 = df2.drop(['Sex'], axis=1)
    dm = pd.get_dummies(df['Embarked'], drop_first=True)
    df2['Embarked'] = df2['Embarked'].fillna('S')
    df3 = pd.concat([df2, dm], axis=1)
    df3 = df3.drop(['Embarked'], axis=1)
    df3['Age'] = df3['Age'].apply(lambda x: math.floor(x))
    return df3

train = pd.read_csv('C:\\Users\\user\\OneDrive\\Documents\\train.csv')
test = pd.read_csv('C:\\Users\\user\\OneDrive\\Documents\\tested.csv')
traindata = clear(train)
testdata = clear(test)

xtrain = traindata.drop(['Survived'], axis=1)
ytrain = traindata.Survived
xtest = testdata.drop(['Survived'], axis=1)
ytest = testdata.Survived

xtest.isnull().sum()
meanfare = np.mean(xtest['Fare'])
xtest['Fare'] = xtest['Fare'].fillna(meanfare)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(xtrain, ytrain)
model.score(xtest, ytest)

ypredict = model.predict(xtest)
sub = pd.DataFrame()
sub['PassengerId'] = xtest['PassengerId']
sub['Survived'] = ypredict
sub.to_csv('gender_submission.csv', index=False)
