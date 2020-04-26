import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# TODO: Create a catplot showing the relation from Pclases to survival
# TODO: Draw correlation between survival and (SibSp || Parch)

def embark(data):
  # Display the count of the 'Embarked' values to show a disparity
  # Results show the majority of passengers embarked from a single location
  # Conclusion: data in relation to Embarked == Q will be less accurate than Embarked == S
  print(data['Embarked'].value_counts() / len(data))
  embarkCountPlot(data)

  # Does gender have an influence over this?
  # Results show inconsistent relationships between gender/survival against embarked
  # Conclusion: supports theory that smaller data sample will yield different results
  meanSex = data.groupby('Sex').mean()
  print(meanSex)
  embarkFacetGrid(data)


def sex(data):
  sexCatplot(data)
  sexAgePlot(data)

def getAgeQuantile(row, percentiles):
  if row['Age'] < percentiles[0]: return 25
  elif row['Age'] < percentiles[1]: return 50
  elif row['Age'] < percentiles[2]: return 75
  else: return 100

def sexAgePlot(d):
  agePercent = np.percentile(d['Age'], [25,50,75,100])
  d['AgeQuantile'] = d.apply(lambda x: getAgeQuantile(x, agePercent), axis=1)
  sns.catplot(x='AgeQuantile', y='Survived', hue='Sex',data=d, kind='bar')
  plt.ylim(0,1)
  plt.show()

def sexCatplot(d):
  sns.catplot(x='Sex', y='Survived', hue='Pclass', data=d, height=6, kind='bar')
  plt.show()

def embarkCountPlot(d):
  sns.countplot( x='Embarked', data=d, hue='Survived', palette='Set2')
  plt.show()

def embarkFacetGrid(data):
  fg = sns.FacetGrid(data, row='Embarked', height=5, aspect=1.6)
  fg.map(sns.pointplot, 'Sex', 'Survived', 'Pclass', order=None, hue_order=None)
  fg.add_legend()
  plt.show()

def heatmap(data):
  plt.figure(figsize=(12,12))
  sns.heatmap(data.corr(), vmax=1, square=True, annot=True)
  plt.show()