import os
import pandas as pd
import numpy as np
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
  return f"{' '.join(data['keywords'])} {' '.join(data['cast'])} {data['director']} {' '.join(data['genres'])}"

def extractRecommendations(title, indexReference, cosSimilarity, movies):
  similarityScores = list(enumerate(cosSimilarity[indexReference[title]]))
  similarityScores = sorted(similarityScores, key=lambda movie : movie[1], reverse=True)
  movieIndexList = [movie[0] for movie in similarityScores[1:11]]
  return movies['title'].iloc[movieIndexList]

def main():
  dirname = os.path.dirname(__file__)

  moviesDf = pd.read_csv(os.path.join(dirname, '..', 'data', 'tmdb_5000_movies.csv'))
  creditsDf = pd.read_csv(os.path.join(dirname, '..', 'data', 'tmdb_5000_credits.csv'))


  moviesDf = pd.concat([moviesDf, creditsDf.drop(columns=['title',  'movie_id'])], axis=1)

  for jsonFeature in ['cast', 'crew', 'keywords',  'genres', 'production_companies']:
    moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(json.loads)
    if jsonFeature == 'crew':
      moviesDf['director'] = moviesDf[jsonFeature].apply(extractDirector)
      moviesDf['director'] = moviesDf['director'].apply(cleanseData)
    else:
      moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(extractList)
      moviesDf[jsonFeature] = moviesDf[jsonFeature].apply(cleanseData)



  cleanMovies = moviesDf[['title', 'cast', 'director', 'keywords', 'genres', 'production_companies']]

  vectors = cleanMovies.apply(extractVector, axis=1)

  count = CountVectorizer(stop_words='english')
  matrix = count.fit_transform(vectors)

  similarity = cosine_similarity(matrix, matrix)
  indexReference = pd.Series(cleanMovies.index, index=cleanMovies['title'])

  print(extractRecommendations('The Dark Knight Rises', indexReference, similarity, cleanMovies))
  print(extractRecommendations('Spy Kids', indexReference, similarity, cleanMovies))

main()


