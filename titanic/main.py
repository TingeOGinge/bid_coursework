import os
import pandas as pd

from titanic.preprocessing import preVisualPreprocessing, postVisualPreprocessing
from titanic import analysis, visualisation


def writeOutput(final, dirname, train, test):
  prediction = analysis.runFinalTest(train, test)
  final['Survived'] = prediction

  final.to_csv(os.path.join(dirname, '..', 'data', 'titanic-output.csv'), index=False)

def main():
  dirname = os.path.dirname(__file__)

  train = pd.read_csv(os.path.join(dirname, '..', 'data', 'train.csv'))
  test = pd.read_csv(os.path.join(dirname, '..', 'data', 'test.csv'))
  final = test[['PassengerId']]

  train = preVisualPreprocessing(train)
  test = preVisualPreprocessing(test)

  visualisationFlag = input("Would you like to see the visualisations? (Y/N)") is "Y"
  if visualisationFlag: visualisation.mainVisuals(train)

  train = postVisualPreprocessing(train)
  test = postVisualPreprocessing(test)

  if visualisationFlag: visualisation.heatmap(train)

  analysis.runBenchmarkTests(train)

  if input("Would you like to write the output? (Y/N)") is 'Y': writeOutput(final, dirname, train, test)

main()