import pandas as pd
import numpy as np
from sklearn import preprocessing
import re

def getAgeQuantile(row, percentiles):
  if row['Age'] < percentiles[0]:
    return 5
  elif row['Age'] < percentiles[1]: return 10
  elif row['Age'] < percentiles[2]: return 25
  elif row['Age'] < percentiles[3]: return 50
  elif row['Age'] < percentiles[4]: return 75
  else: return 100

def stripTitle(row):
  pattern = r'^.+, ([^\.]+)'
  result = re.findall(pattern, row['Name'])

  if len(result) > 0: return result[0] if result[0] in ['Mr', 'Mrs', 'Miss', 'Master'] else 'Outlier'

  return 'N/A'

def preVisualPreprocessing(data):
  data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())
  data['Fare'] = data['Fare'].fillna(data['Fare'].median())

  data['Age'] = data['Age'].fillna(data['Age'].median())
  agePercent = np.percentile(data['Age'], [5, 10, 25, 50, 75, 100])
  print(f"Age Percenitile Range: {agePercent}")
  data['AgeQuantile'] = data.apply(lambda row: getAgeQuantile(row, agePercent), axis=1)

  data['Relatives'] = data['SibSp'] + data['Parch']

  data['Title'] = data.apply(stripTitle, axis=1)

  return data

def postVisualPreprocessing(data):
  data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

  minmaxScaler = preprocessing.MinMaxScaler()

  scalerList = ['Fare', 'Age', 'SibSp', 'Parch']

  for scale in scalerList: data[scale] = minmaxScaler.fit_transform(data[[scale]].values)

  data['FarePerPerson'] = data['Fare'] / (data['Relatives'] + 1)

  data['Maturity'] = data.apply(lambda row: "child" if row['AgeQuantile'] == 5 else 'Adult', axis=1)

  data = data.drop(['Cabin','Name','Ticket','AgeQuantile','Embarked','Age','SibSp','Parch','PassengerId'], axis=1)

  data['Relatives'] = data.apply(lambda row: str(row['Relatives']) if row['Relatives'] < 4 else '4+', axis=1)

  encodeList = ['Pclass', 'Sex', 'Title', 'Relatives', 'Maturity']
  for encoding in encodeList:
    data = pd.concat([data, pd.get_dummies(data[encoding], prefix=encoding)], axis=1)
    data = data.drop(encoding, axis=1)

  return data
