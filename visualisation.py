import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# TODO: Visualise correlations relating to title feature (create feature first and use heatmap for inspiration)

def mainVisuals(data):

  pclassBarplot(data)

  print(data['Embarked'].value_counts() / len(data))
  embarkCountPlot(data)

  titleCountPlot(data)

  meanSex = data.groupby('Sex').mean()
  print(meanSex)

  sexCatbarplot(data)
  embarkFacetGrid(data)
  sexAgePointplot(data)

  relativesPointplot(data)

def titleCountPlot(d):
  sns.catplot(x='Title', hue='Survived', data=d, palette='Set1', kind='count', aspect=2.5)
  plt.show()

def relativesPointplot(d):
  sns.catplot('Relatives', 'Survived', hue='Sex', data=d, aspect=2.5, kind='point')
  plt.show()

def pclassBarplot(d):
  sns.barplot(x='Pclass', y='Survived', data=d)
  plt.show()

def sexAgePointplot(d):
  sns.catplot(x='AgeQuantile', y='Survived', hue='Sex',data=d, kind='point')
  plt.ylim(0,1)
  plt.show()

def sexCatbarplot(d):
  sns.catplot(x='Sex', y='Survived', hue='Pclass', data=d, height=6, kind='bar')
  plt.show()

def embarkCountPlot(d):
  sns.countplot( x='Embarked', data=d, hue='Survived', palette='Set2')
  plt.show()

def embarkFacetGrid(data):
  fg = sns.FacetGrid(data, row='Embarked', height=5, aspect=1.6)
  fg.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None)
  fg.add_legend()
  plt.show()

def heatmap(data):
  plt.figure(figsize=(24,24))
  sns.heatmap(data.corr(), vmax=1, square=True, annot=True, fmt='.2f')
  plt.show()

def confusionM(y_true,y_predict,target_names):
  cMatrix = confusion_matrix(y_true,y_predict)
  df_cm = pd.DataFrame(cMatrix,index=target_names,columns=target_names)
  plt.figure(figsize = (6,4))
  cm = sns.heatmap(df_cm,annot=True,fmt="2.0f")
  cm.yaxis.set_ticklabels(cm.yaxis.get_ticklabels(),rotation=90)
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()