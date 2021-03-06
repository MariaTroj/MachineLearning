import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    movies_df = pd.read_csv('..\\ml-latest\\movies.csv')
    ratings_df = pd.read_csv('..\\ml-latest\\ratings.csv')
    ratings_df = ratings_df.drop('timestamp', 1)
    # Preprocess movied_df
    # extract year from movie description and remove year from description
    movies_df['year'] = movies_df.title.str.extract('\((\d\d\d\d)\)',expand=False, regex=True)
    movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '', regex=True)
    movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())
    movies_df['genres'] = movies_df.genres.str.split('|')

    # expand movieWithGenres_df by genres columns
    moviesWithGenres_df = movies_df.copy()
    for index, row in movies_df.iterrows():
        for genre in row['genres']:
            moviesWithGenres_df.at[index, genre] = 1
    moviesWithGenres_df = moviesWithGenres_df.fillna(0)

    # prepare user profile
    userInput = [
        {'title': 'Breakfast Club, The', 'rating': 5},
        {'title': 'Toy Story', 'rating': 3.5},
        {'title': 'Jumanji', 'rating': 2},
        {'title': "Pulp Fiction", 'rating': 5},
        {'title': 'Akira', 'rating': 4.5}
    ]
    inputMovies = pd.DataFrame(userInput)

    # Filter out  movies id rated by user from movies_df and merge them to inputMovies
    inputId = movies_df[movies_df['title'].isin(inputMovies['title'].tolist())]
    inputMovies = pd.merge(inputId, inputMovies)
    inputMovies = inputMovies.drop('genres', 1).drop('year', 1)

    # Filter out the movies from the moviesWithGenres_df
    userMovies = moviesWithGenres_df[
        moviesWithGenres_df['movieId'].isin(inputMovies['movieId'].tolist())]

    # Reset the index to avoid future issues and drop unnecesary columns
    userMovies = userMovies.reset_index(drop=True)
    userGenreTable = userMovies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year',1)
    userProfile = userGenreTable.transpose().dot(inputMovies['rating'])

    # Get the genres of every movie in original dataframe and drop unnecesary columns
    genreTable = moviesWithGenres_df.set_index(moviesWithGenres_df['movieId'])
    genreTable = genreTable.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

    # Multiply the genres by the weights and then take the weighted average
    recommendationTable_df = ((genreTable * userProfile).sum(axis=1)) / (userProfile.sum())

    # Sort our recommendations in descending order
    recommendationTable_df = recommendationTable_df.sort_values(ascending=False)

    # The final recommendation table
    movies_df.loc[movies_df['movieId'].isin(recommendationTable_df.head(20).keys())]