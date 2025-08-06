from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def get_top_n_similar(tweet_vec, movie_vectors, movie_df, n=10):
    """Returns top N similar movies based on cosine similarity"""
    sims = cosine_similarity([tweet_vec], movie_vectors)[0]
    movie_df = movie_df.copy()
    movie_df['similarity'] = sims
    return movie_df.sort_values(by='similarity', ascending=False).head(n)
