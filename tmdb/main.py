import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tmdb.preprocessing import cleanMovieData

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

  cleanMovies, vectors = cleanMovieData(moviesDf)
  count = CountVectorizer(stop_words='english')
  matrix = count.fit_transform(vectors)
  similarity = cosine_similarity(matrix, matrix)
  indexReference = pd.Series(cleanMovies.index, index=cleanMovies['title'])

  print(extractRecommendations('Jurassic World', indexReference, similarity, cleanMovies))
  print(extractRecommendations('Monsters University', indexReference, similarity, cleanMovies))

main()


