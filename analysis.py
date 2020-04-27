import visualisation as vis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# TODO: Reseach classification models and apply a wider range to data

def applyModel(model, X_train, y_train, X_test, y_true, features, targetValues):
  model.fit(X_train, y_train)
  modelPrediction = model.predict(X_test)

  modelScore = round(accuracy_score(y_true, modelPrediction) * 100, 5)
  modelXValScore = round(cross_val_score(model, features, targetValues, cv=10, scoring='accuracy').mean() * 100, 5)
  modelXValPredict = cross_val_predict(model, features, targetValues, cv=10)

  return modelScore, modelXValScore, modelXValPredict

def outputResults(name, base, xvalidScore, xvalidPredict, targetValues, predictionLabels):
  print(f"{name}: \nBase accuracy: {base} \nCross validated score: {xvalidScore}\n")
  vis.confusionM(targetValues, xvalidPredict, predictionLabels)

def linearDA(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  lda = LinearDiscriminantAnalysis()
  ldaScore, ldaXValScore, ldaXValPredict = applyModel(lda, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Linear Discrimination Analysis', ldaScore, ldaXValScore, ldaXValPredict, targetValues, predictionLabels)

def logisticRegr(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  logReg = LogisticRegression()
  logRegScore, logRegXValScore, logRegXValPredict = applyModel(logReg, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Logistic Regression', logRegScore, logRegXValScore, logRegXValPredict, targetValues, predictionLabels)

def randomForestClass(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  rfClass = RandomForestClassifier(max_depth=5)
  rfClassScore, rfClassXValScore, rfClassXValPredict = applyModel(rfClass, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Random Forest Classifier', rfClassScore, rfClassXValScore, rfClassXValPredict, targetValues, predictionLabels)

def runTests(data):
  targetValues = data['Survived']
  features = data.drop('Survived', axis=1)
  predictionLabels = data['Survived'].unique()
  X_train, X_test, y_train, y_true = train_test_split(features, targetValues)

  for f in [linearDA, logisticRegr, randomForestClass]:
    f(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels)