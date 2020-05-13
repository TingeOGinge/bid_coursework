import visualisation as vis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def applyModel(model, X_train, y_train, X_test, y_true, features, targetValues):
  model.fit(X_train, y_train)
  modelPrediction = model.predict(X_test)

  modelScore = round(accuracy_score(y_true, modelPrediction) * 100, 5)
  modelXValScore = round(cross_val_score(model, features, targetValues, cv=10, scoring='accuracy').mean() * 100, 5)
  modelXValPredict = cross_val_predict(model, features, targetValues, cv=10)

  return modelScore, modelXValScore, modelXValPredict

def outputResults(name, base, xvalidScore, xvalidPredict, targetValues, predictionLabels):
  print(f"{name}: \nBase accuracy: {base} \nCross validated score: {xvalidScore}\n")
  vis.confusionM(targetValues, xvalidPredict, predictionLabels, name)

def linearDA(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  lda = LinearDiscriminantAnalysis()
  ldaScore, ldaXValScore, ldaXValPredict = applyModel(lda, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Linear Discrimination Analysis', ldaScore, ldaXValScore, ldaXValPredict, targetValues, predictionLabels)

def logisticRegr(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  logReg = LogisticRegression()
  logRegScore, logRegXValScore, logRegXValPredict = applyModel(logReg, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Logistic Regression', logRegScore, logRegXValScore, logRegXValPredict, targetValues, predictionLabels)

def randomForestClass(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  rfClass = RandomForestClassifier(max_depth=10, random_state=3)
  rfClassScore, rfClassXValScore, rfClassXValPredict = applyModel(rfClass, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Random Forest Classifier', rfClassScore, rfClassXValScore, rfClassXValPredict, targetValues, predictionLabels)

def kNN(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  knn = KNeighborsClassifier()
  knnScore, knnXValScore, knnXValPredict = applyModel(knn, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('K Nearest Neighbour', knnScore, knnXValScore, knnXValPredict, targetValues, predictionLabels)

def sVC(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  svc = SVC(gamma='auto')
  svcScore, svcXValScore, svcXValPredict = applyModel(svc, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Support Vector Classification', svcScore, svcXValScore, svcXValPredict, targetValues, predictionLabels)

def decisionTreeClass(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels):
  dcClass = DecisionTreeClassifier()
  dcClassScore, dcClassXValScore, dcClassXValPredict = applyModel(dcClass, X_train, y_train, X_test, y_true, features, targetValues)
  outputResults('Decision Tree Classifier', dcClassScore, dcClassXValScore, dcClassXValPredict, targetValues, predictionLabels)

def runBenchmarkTests(data):
  targetValues = data['Survived']
  features = data.drop('Survived', axis=1)
  predictionLabels = data['Survived'].unique()
  X_train, X_test, y_train, y_true = train_test_split(features, targetValues)


  for f in [linearDA, logisticRegr, decisionTreeClass, randomForestClass, kNN, sVC]:
    f(X_train, y_train, X_test, y_true, targetValues, features, predictionLabels)

def runFinalTest(train, test):
  finalModel = RandomForestClassifier(max_depth=10, random_state=3)
  finalModel.fit(train.drop(['Survived'], axis=1), train['Survived'])
  return finalModel.predict(test)


