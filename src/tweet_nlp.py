# First, ensure you have the library installed:
# pip install transformers torch

from transformers import pipeline
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_thought_to_genres(thought_text, threshold=0.3, return_scores=False):
    """
    Convert natural language movie descriptions into genre predictions using zero-shot classification.
    
    Args:
        thought_text (str): Natural language description of desired movie
        threshold (float): Confidence threshold for genre inclusion (0.0 to 1.0)
        return_scores (bool): If True, returns scores along with genres
    
    Returns:
        list: Predicted genres, optionally with confidence scores
    """
    
    # Initialize the zero-shot classification pipeline
    try:
        classifier = pipeline(
            "zero-shot-classification", 
            model="facebook/bart-large-mnli",
            device=0 if hasattr(pipeline, 'device') else -1  # Use GPU if available
        )
    except Exception as e:
        logger.warning(f"Falling back to CPU: {e}")
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Extended list of movie genres including sub-genres and hybrid genres
    candidate_genres = [
        # Core Genres
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Fantasy', 'History', 'Horror', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western',
        
        # Sub-genres and Hybrid Genres
        'Romantic Comedy', 'Action Comedy', 'Horror Comedy', 'Sci-Fi Horror',
        'Action Thriller', 'Crime Thriller', 'Psychological Thriller',
        'Supernatural Horror', 'Slasher Horror', 'Psychological Horror',
        'Historical Drama', 'War Drama', 'Crime Drama', 'Legal Drama',
        'Superhero Action', 'Martial Arts', 'Heist Film', 'Spy Film',
        'Disaster Film', 'Mockumentary', 'Biographical Drama',
        
        # Mood-based Genres
        'Dark Comedy', 'Black Comedy', 'Satire', 'Parody',
        'Epic', 'Noir', 'Gothic', 'Melodrama', 'Tragicomedy',
        
        # Niche/Specific Genres
        'Musical', 'Sports Drama', 'Teen Comedy', 'Coming of Age',
        'Family Film', 'Children\'s Film', 'Anime', 'Silent Film',
        'Road Movie', 'Courtroom Drama', 'Political Thriller'
    ]

    # Validate input
    if not thought_text or not isinstance(thought_text, str):
        logger.warning("Invalid input: thought_text must be a non-empty string")
        return [] if not return_scores else ([], {})

    try:
        # Perform zero-shot classification
        logger.info("Analyzing thought for genre prediction...")
        result = classifier(thought_text, candidate_genres)
        
        # Filter results based on threshold
        predicted_genres = []
        genre_scores = {}
        
        for label, score in zip(result['labels'], result['scores']):
            if score > threshold:
                predicted_genres.append(label)
                genre_scores[label] = round(score, 3)
        
        # If no genres meet threshold, return top 2-3 genres
        if not predicted_genres and result['scores']:
            top_indices = sorted(range(len(result['scores'])), 
                               key=lambda i: result['scores'][i], reverse=True)[:3]
            predicted_genres = [result['labels'][i] for i in top_indices]
            genre_scores = {result['labels'][i]: round(result['scores'][i], 3) 
                          for i in top_indices}
        
        logger.info(f"Predicted genres: {predicted_genres}")
        
        if return_scores:
            return predicted_genres, genre_scores
        else:
            return predicted_genres
            
    except Exception as e:
        logger.error(f"Error in genre prediction: {e}")
        return [] if not return_scores else ([], {})

def get_genre_confidence_scores(thought_text, threshold=0.3):
    """
    Convenience function to get both genres and their confidence scores.
    
    Args:
        thought_text (str): Natural language description of desired movie
        threshold (float): Confidence threshold for genre inclusion
    
    Returns:
        tuple: (list of genres, dict of genre:score mappings)
    """
    return convert_thought_to_genres(thought_text, threshold, return_scores=True)

# --- Enhanced Example Usage ---

if __name__ == "__main__":
    # Example 1: A thought about a futuristic action movie
    thought1 = "I want to see something with spaceships, laser battles, and a lone hero fighting against a galactic empire."
    genres1 = convert_thought_to_genres(thought1)
    genres1_with_scores, scores1 = get_genre_confidence_scores(thought1)
    
    print("🎬 Example 1: Futuristic Action Movie")
    print(f"Thought: \"{thought1}\"")
    print(f"Predicted Genres: {genres1}")
    print(f"Genres with Scores: {dict(list(scores1.items())[:5])}")  # Top 5
    print("-" * 50)

    # Example 2: A thought about a light-hearted story
    thought2 = "I'm in the mood for something funny and sweet, maybe about two people falling in love against the odds."
    genres2 = convert_thought_to_genres(thought2)
    genres2_with_scores, scores2 = get_genre_confidence_scores(thought2)
    
    print("💕 Example 2: Romantic Comedy")
    print(f"Thought: \"{thought2}\"")
    print(f"Predicted Genres: {genres2}")
    print(f"Genres with Scores: {dict(list(scores2.items())[:5])}")  # Top 5
    print("-" * 50)

    # Example 3: A thought about a tense, scary movie
    thought3 = "I want to watch a movie that will keep me on the edge of my seat, with a shocking twist at the end and maybe a ghost."
    genres3 = convert_thought_to_genres(thought3)
    genres3_with_scores, scores3 = get_genre_confidence_scores(thought3)
    
    print("👻 Example 3: Psychological Horror/Thriller")
    print(f"Thought: \"{thought3}\"")
    print(f"Predicted Genres: {genres3}")
    print(f"Genres with Scores: {dict(list(scores3.items())[:5])}")  # Top 5
    print("-" * 50)

    # Example 4: Complex hybrid genre
    thought4 = "I want to see martial arts action mixed with comedy, like kung fu but hilarious."
    genres4 = convert_thought_to_genres(thought4, threshold=0.25)
    genres4_with_scores, scores4 = get_genre_confidence_scores(thought4, threshold=0.25)
    
    print("🥋 Example 4: Martial Arts Comedy")
    print(f"Thought: \"{thought4}\"")
    print(f"Predicted Genres: {genres4}")
    print(f"Genres with Scores: {dict(list(scores4.items())[:5])}")  # Top 5
    print("-" * 50)

    # Example 5: Niche genre
    thought5 = "I'm looking for a mockumentary style comedy about everyday office life."
    genres5 = convert_thought_to_genres(thought5)
    genres5_with_scores, scores5 = get_genre_confidence_scores(thought5)
    
    print("🏢 Example 5: Mockumentary Comedy")
    print(f"Thought: \"{thought5}\"")
    print(f"Predicted Genres: {genres5}")
    print(f"Genres with Scores: {dict(list(scores5.items())[:5])}")  # Top 5