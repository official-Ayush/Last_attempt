# First, ensure you have the library installed:
# pip install transformers

from transformers import pipeline

def convert_thought_to_genres(thought_text, threshold=0.4):

    classifier = pipeline("zero-shot-classification", 
                          model="facebook/bart-large-mnli")

    # 2. Define the Candidate Genres
    # These are the labels the model will choose from.
    candidate_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Fantasy', 'History', 'Horror', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    # 3. Classify the Thought
    # The model calculates a probability score for each genre.
    result = classifier(thought_text, candidate_genres)

    # 4. Filter Results Based on a Threshold
    # We only keep the genres that the model is confident about.
    predicted_genres = []
    for i in range(len(result['labels'])):
        if result['scores'][i] > threshold:
            predicted_genres.append(result['labels'][i])
            
    return predicted_genres

# --- Example Usage ---

# Example 1: A thought about a futuristic action movie
thought1 = "I want to see something with spaceships, laser battles, and a lone hero fighting against a galactic empire."
genres1 = convert_thought_to_genres(thought1)
print(f"Thought: \"{thought1}\"")
print(f"Predicted Genres: {genres1}")
# Expected Output: ['Sci-Fi', 'Action', 'Adventure']

print("-" * 20)

# Example 2: A thought about a light-hearted story
thought2 = "I'm in the mood for something funny and sweet, maybe about two people falling in love against the odds."
genres2 = convert_thought_to_genres(thought2)
print(f"Thought: \"{thought2}\"")
print(f"Predicted Genres: {genres2}")
# Expected Output: ['Romance', 'Comedy', 'Drama']

print("-" * 20)

# Example 3: A thought about a tense, scary movie
thought3 = "I want to watch a movie that will keep me on the edge of my seat, with a shocking twist at the end and maybe a ghost."
genres3 = convert_thought_to_genres(thought3)
print(f"Thought: \"{thought3}\"")
print(f"Predicted Genres: {genres3}")
# Expected Output: ['Thriller', 'Mystery', 'Horror']