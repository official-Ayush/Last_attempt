import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

def recommend_engine(predicted_genres):
    """
    Loads the trained recommender model and predicts a movie title
    based on a list of input genres.

    Args:
        predicted_genres (list): A list of genre strings, e.g., ['Horror', 'Thriller'].

    Returns:
        str: The title of the recommended movie.
    """
    # --- 1. Load the Pre-trained Model ---
    # Construct the path to the model file relative to this script's location.
    # This makes the code more portable.
    try:
        # Assuming your script is in src/ and your model is in models/
        # This builds the correct path: D:\Boring_Project1\models\genre_to_title_model.pkl
        script_dir = os.path.dirname(__file__) # The directory of this script (src)
        project_root = os.path.dirname(script_dir) # The directory above src (Boring_Project1)
        model_path = os.path.join(project_root, 'models', 'genre_to_title_model.pkl')
        
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
    except FileNotFoundError:
        return "Error: Model file not found. Please run the training script first."

    # --- 2. Extract Components from the Loaded Model ---
    vectorizer = model_components['vectorizer']
    genre_matrix = model_components['genre_matrix']
    id_to_title_map = model_components['id_to_title']

    # --- 3. Process the Input Genres ---
    # Join the list of genres into a single string for the vectorizer.
    input_string = ' '.join(predicted_genres)
    
    # Transform the input string into a numerical vector using the loaded vectorizer.
    input_vector = vectorizer.transform([input_string])

    # --- 4. Find the Best Match ---
    # Calculate the cosine similarity between the input vector and all movie vectors.
    cosine_similarities = cosine_similarity(input_vector, genre_matrix).flatten()
    
    # Find the index of the movie with the highest similarity score.
    most_similar_movie_index = np.argmax(cosine_similarities)
    
    # --- 5. Return the Predicted Title ---
    # Use the index to look up the movie title from the mapping.
    predicted_title = id_to_title_map.iloc[most_similar_movie_index]
    
    return predicted_title

# --- Example of how to use this function ---
if __name__ == '__main__':
    # Test Case 1: Horror/Thriller
    genres1 = ['Horror', 'Thriller', 'Mystery']
    title1 = recommend_engine(genres1)
    print(f"Input Genres: {genres1}")
    print(f"Recommended Title: {title1}")

    print("-" * 20)

    # Test Case 2: Action/Adventure
    genres2 = ['Action', 'Adventure']
    title2 = recommend_engine(genres2)
    print(f"Input Genres: {genres2}")
    print(f"Recommended Title: {title2}")