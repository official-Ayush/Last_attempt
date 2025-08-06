import pandas as pd

def load_and_preprocess_data(movies_path = 'D:/Boring_Project1/data/movies.csv', ratings_path =r'D:/Boring_Project1/data/ratings.csv'):
    """Loads, merges, and cleans the movie and rating datasets."""
    movies = pd.read_csv(movies_path) 
    movies["genres"] = movies["genres"].str.split("|")
    movies_explode = movies.explode("genres")

    ratings = pd.read_csv(ratings_path)
# Since the timestamp is in seconds we will use datetime method from pandas to convert it.
    ratings["date"] = pd.to_datetime(ratings["timestamp"], unit='s') 
    ratings = ratings.drop(["timestamp"], axis=1)  # dropping the datetime column for uncessary changes 
    ratings["date"] = ratings["date"].dt.date
    
    df = pd.merge(ratings, movies_explode)
    
    # ... add your other cleaning steps from EDA here ...
    # (e.g., creating a 'year' column, handling missing values)
    df['year'] = df['title'].str.extract(r'\((\d{4})\)')
    
    return df