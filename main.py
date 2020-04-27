import os
import visualisation as vis
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def getAgeQuantile(row, percentiles):
  if row['Age'] < percentiles[0]: return 25
  elif row['Age'] < percentiles[1]: return 50
  elif row['Age'] < percentiles[2]: return 75
  else: return 100

def preVisualPreprocessing(data):
  data = data.set_index(['PassengerId'])
  data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode())

  data['Age'] = data['Age'].fillna(data['Age'].median())
  agePercent = np.percentile(data['Age'], [5, 10, 25, 50, 75, 100])
  data['AgeQuantile'] = data.apply(lambda row: getAgeQuantile(row, agePercent), axis=1)

  data['Relatives'] = data['SibSp'] + data['Parch']
  data['SoloTraveller'] = data.apply(lambda row: 'No' if row['Relatives'] > 0 else 'Yes', axis=1)

  return data

def postVisualPreprocessing(data):
  data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

  minmaxScaler = preprocessing.MinMaxScaler()
  fareAsFloat = data[['Fare']].values.astype(float)

  data['Fare'] = minmaxScaler.fit_transform(fareAsFloat)
  data['Age'] = minmaxScaler.fit_transform(data[['Age']].values)
  data['SibSp'] = minmaxScaler.fit_transform(data[['SibSp']].values)
  data['Parch'] = minmaxScaler.fit_transform(data[['Parch']].values)

  data['FarePerPerson'] = data['Fare'] / (data['Relatives']+1)

  data = data.drop(['Cabin', 'Name', 'Ticket'], axis=1)

  encodeList = ['AgeQuantile', 'Pclass', 'Embarked', 'SoloTraveller']
  for encoding in encodeList:
    data = pd.concat([data, pd.get_dummies(data[encoding], prefix=encoding)], axis=1)
    data = data.drop(encoding, axis=1)

  return data

def main():
  dirname = os.path.dirname(__file__)

  train = pd.read_csv(os.path.join(dirname, 'data', 'train.csv'))
  test = pd.read_csv(os.path.join(dirname, 'data', 'test.csv'))

  train = preVisualPreprocessing(train)
  test = preVisualPreprocessing(test)

  vis.mainVisuals(train)

  train = postVisualPreprocessing(train)
  test = postVisualPreprocessing(test)

  vis.heatmap(train)

  X_train, X_test, y_train, y_true = train_test_split(train.drop('Survived', axis=1), train['Survived'])
  lda = LinearDiscriminantAnalysis()
  lda.fit(X_train, y_train)
  y_predict = lda.predict(X_test)

  vis.confusionM(y_true,y_predict,train['Survived'].unique())
  print(f"Accuracy {round(accuracy_score(y_true, y_predict) * 100, 5):}")

  return 0

main()