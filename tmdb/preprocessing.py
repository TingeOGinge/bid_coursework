import json
import numpy as np

def extractDirector(crew):
  for member in crew:
    if member['job'] == 'Director': return member['name']

  return np.nan

def extractList(obj):
  try:
    nameList = [x['name'] for x in obj]
    return nameList[:4] if len(nameList) > 4 else nameList
  except TypeError:
    return []

def cleanseData(data):
  try:
    if isinstance(data, list):
      return [str.lower(x.replace(" ", "")) for x in data]
    else:
      return data.replace(" ", "").lower()
  except AttributeError:
    return ''

def extractVector(data):
  return f"{' '.join(data['keywords'])} {' '.join(data['cast'])} {(data['director'] + ' ') * 2} {' '.join(data['genres'])}"

def cleanMovieData(moviesDf):
  for jsonFeature in ['cast', 'crew', 'keywords', 'genres', 'production_companies']:
    moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(json.loads)
    if jsonFeature == 'crew':
      moviesDf['director'] = moviesDf[jsonFeature].apply(extractDirector)
      moviesDf['director'] = moviesDf['director'].apply(cleanseData)
    else:
      moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(extractList)
      moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(cleanseData)

  cleanMovies =  moviesDf[['title', 'cast', 'director', 'keywords', 'genres', 'production_companies']]
  return cleanMovies, cleanMovies.apply(extractVector, axis=1)
