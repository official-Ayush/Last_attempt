import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recommend_engine(predicted_genres, top_k=5, strict_genre_match=True):
    """
    Recommend movies based on input genres.

    Args:
        predicted_genres (list): List of genre strings.
        top_k (int): Number of recommendations to return.
        strict_genre_match (bool): If True, only recommend movies with at least one matching genre.

    Returns:
        list: Titles of recommended movies.
    """
    # --- 1. Load Model ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        model_path = os.path.join(project_root, 'models', 'genre_to_title_model.pkl')
        with open(model_path, 'rb') as f:
            model_components = pickle.load(f)
    except FileNotFoundError:
        return ["Error: Model file not found. Please run the training script first."]
    except Exception as e:
        return [f"Error loading model: {str(e)}"]

    vectorizer = model_components['vectorizer']
    genre_matrix = model_components['genre_matrix']
    id_to_title_map = model_components['id_to_title']

    # --- 2. Validate Input Genres ---
    all_genres = set(vectorizer.get_feature_names_out())
    unknown_genres = set(predicted_genres) - all_genres
    if unknown_genres:
        logger.warning(f"Unknown genres: {unknown_genres}")

    known_genres = [g for g in predicted_genres if g in all_genres]
    
    if not known_genres:
        logger.warning("No valid genres found after filtering")
        # Fallback: return random popular movies
        indices = np.random.choice(len(id_to_title_map), min(top_k, len(id_to_title_map)), replace=False)
        return [id_to_title_map.iloc[i] for i in indices]

    # --- 3. Build Input Vector ---
    input_vector = np.zeros((1, len(all_genres)))
    for genre in known_genres:
        try:
            idx = list(all_genres).index(genre)
            input_vector[0, idx] = 1
        except ValueError:
            logger.warning(f"Genre {genre} not found in vectorizer features")

    # --- 4. Genre Matching Filter ---
    valid_indices = np.arange(len(genre_matrix))
    filtered_genre_matrix = genre_matrix
    
    if strict_genre_match:
        try:
            genre_columns = vectorizer.get_feature_names_out()
            input_mask = np.array([genre in known_genres for genre in genre_columns])
            
            # Check if any genres match
            matching_rows_mask = genre_matrix[:, input_mask].sum(axis=1) > 0
            valid_indices = np.where(matching_rows_mask)[0]
            
            if len(valid_indices) > 0:
                filtered_genre_matrix = genre_matrix[valid_indices]
                logger.info(f"Found {len(valid_indices)} movies matching the genres")
            else:
                logger.warning("No movies found with matching genres, falling back to all movies")
                # Fallback to all movies when no matches found
                valid_indices = np.arange(len(genre_matrix))
                filtered_genre_matrix = genre_matrix
                
        except Exception as e:
            logger.error(f"Error in genre filtering: {e}")
            # Fallback to all movies
            valid_indices = np.arange(len(genre_matrix))
            filtered_genre_matrix = genre_matrix

    # --- 5. Compute Similarity ---
    try:
        if filtered_genre_matrix.shape[0] == 0:
            logger.error("Filtered genre matrix is empty")
            # Ultimate fallback
            indices = np.random.choice(len(id_to_title_map), min(top_k, len(id_to_title_map)), replace=False)
            return [id_to_title_map.iloc[i] for i in indices]
            
        cosine_similarities = cosine_similarity(input_vector, filtered_genre_matrix).flatten()
        
        # Handle case where all similarities are the same (e.g., all zeros)
        if np.all(cosine_similarities == cosine_similarities[0]):
            logger.warning("All cosine similarities are equal, returning random selection")
            top_indices_in_filtered = np.random.choice(
                len(valid_indices), 
                size=min(top_k, len(valid_indices)), 
                replace=False
            )
        else:
            top_indices_in_filtered = np.argsort(cosine_similarities)[::-1][:top_k]
            
        top_indices = valid_indices[top_indices_in_filtered]
        
    except Exception as e:
        logger.error(f"Error computing similarities: {e}")
        # Final fallback
        indices = np.random.choice(len(id_to_title_map), min(top_k, len(id_to_title_map)), replace=False)
        return [id_to_title_map.iloc[i] for i in indices]

    # --- 6. Return Titles ---
    try:
        recommended_titles = [id_to_title_map.iloc[i] for i in top_indices]
        return recommended_titles if recommended_titles else ["No recommendations found"]
    except Exception as e:
        logger.error(f"Error retrieving titles: {e}")
        return ["Error retrieving movie titles"]

# --- Example Usage ---
if __name__ == '__main__':
    # Test the function
    test_genres = ['Action', 'Adventure']
    recommendations = recommend_engine(test_genres, top_k=3)
    print(f"Recommendations for {test_genres}: {recommendations}")