import pandas as pd
import numpy as np
from sklearn import preprocessing

def getAgeQuantile(row, percentiles):
  if row['Age'] < percentiles[0]: return 5
  elif row['Age'] < percentiles[1]: return 10
  elif row['Age'] < percentiles[2]: return 25
  elif row['Age'] < percentiles[3]: return 50
  elif row['Age'] < percentiles[4]: return 75
  else: return 100

def preVisualPreprocessing(data):
  data = data.set_index(['PassengerId'])
  data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())

  data['Age'] = data['Age'].fillna(data['Age'].median())
  agePercent = np.percentile(data['Age'], [5, 10, 25, 50, 75, 100])
  print(f"Age Percnitile Range: {agePercent}")
  data['AgeQuantile'] = data.apply(lambda row: getAgeQuantile(row, agePercent), axis=1)

  data['Relatives'] = data['SibSp'] + data['Parch']
  data['SoloTraveller'] = data.apply(lambda row: 'No' if row['Relatives'] > 0 else 'Yes', axis=1)

  return data

def postVisualPreprocessing(data):
  data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

  minmaxScaler = preprocessing.MinMaxScaler()

  scalerList = ['Fare', 'Age', 'SibSp', 'Parch']
  for scale in scalerList: data[scale] = minmaxScaler.fit_transform(data[[scale]].values)

  data['FarePerPerson'] = data['Fare'] / (data['Relatives'] + 1)

  data = data.drop(['Cabin', 'Name', 'Ticket', 'Embarked', 'Relatives'], axis=1)

  encodeList = ['AgeQuantile', 'Pclass', 'SoloTraveller']
  for encoding in encodeList:
    data = pd.concat([data, pd.get_dummies(data[encoding], prefix=encoding)], axis=1)
    data = data.drop(encoding, axis=1)

  return data