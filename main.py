import visualisation as vis
import pandas as pd
from sklearn import preprocessing
import os

def main():
  dirname = os.path.dirname(__file__)

  train = pd.read_csv(os.path.join(dirname, 'data', 'train.csv'))
  test = pd.read_csv(os.path.join(dirname, 'data', 'train.csv'))

  train = train.set_index(['PassengerId'])
  train = train.drop(['Cabin', 'Name', 'Ticket'], axis=1)
  train['Age'] = train['Age'].fillna(train['Age'].median())

  vis.sex(train)
  vis.embark(train)

  train['Embarked'] = train['Embarked'].map({'Q': 2, 'S': 1, 'C': 0})
  train['Embarked'] = train['Embarked'].fillna(train['Embarked'].median())
  train['Embarked'] = train['Embarked'].astype(int)

  train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
  minmaxScaler = preprocessing.MinMaxScaler()
  x = train[['Fare']].values.astype(float)
  train['Fare'] = minmaxScaler.fit_transform(x)

  train = pd.concat([train, pd.get_dummies(train['Pclass'], prefix='SES')], axis=1)
  train = train.drop(['Pclass'], axis=1)

  train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix='Embarked')], axis=1)
  train = train.drop(['Embarked'], axis=1)

  vis.heatmap(train)


  return 0

main()