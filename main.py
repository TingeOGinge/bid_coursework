import os
import pandas as pd

import visualisation as visuals
import analysis
from preprocessing import preVisualPreprocessing, postVisualPreprocessing

def main():
  dirname = os.path.dirname(__file__)

  train = pd.read_csv(os.path.join(dirname, 'data', 'train.csv'))
  test = pd.read_csv(os.path.join(dirname, 'data', 'test.csv'))

  train = preVisualPreprocessing(train)
  test = preVisualPreprocessing(test)

  visuals.mainVisuals(train)

  train = postVisualPreprocessing(train)
  test = postVisualPreprocessing(test)

  visuals.heatmap(train)

  analysis.runTests(train)

  return 0

main()