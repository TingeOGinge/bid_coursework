import visualisation as vis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def linearDA(X_train, y_train, X_test):
  lda = LinearDiscriminantAnalysis()
  lda.fit(X_train, y_train)
  return lda.predict(X_test)

def runAnalysis(y_true, y_predict, predictionLabels):
  vis.confusionM(y_true, y_predict, predictionLabels)
  result = round(accuracy_score(y_true, y_predict) * 100, 5)
  print(f"Accuracy {result}")

def runTests(data):
  X_train, X_test, y_train, y_true = train_test_split(data.drop('Survived', axis=1), data['Survived'])
  predictionLabels = data['Survived'].unique()
  y_predict = linearDA(X_train, y_train, X_test)

  runAnalysis(y_true, y_predict, predictionLabels)
