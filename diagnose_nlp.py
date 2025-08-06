# diagnose_nlp.py
from transformers import pipeline
import os

# This helps prevent some network errors on certain systems
os.environ['CURL_CA_BUNDLE'] = ''

print("Attempting to initialize the NLP pipeline...")
print("This may take a while as it will download a 1.6 GB model if not cached.")

try:
    # Initialize the classifier
    classifier = pipeline("zero-shot-classification",
                          model="facebook/bart-large-mnli")

    print("\nModel initialized successfully!")

    # Define a simple thought and the candidate genres
    thought_text = "I want to watch a movie about space travel and aliens."
    candidate_genres = [
        'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Fantasy', 'History', 'Horror', 'Mystery', 'Romance',
        'Sci-Fi', 'Thriller', 'War', 'Western'
    ]

    # Get the raw results from the model
    results = classifier(thought_text, candidate_genres)

    # Print the full, unfiltered results
    print("\n--- RAW MODEL OUTPUT ---")
    print(f"Thought: '{results['sequence']}'")
    for i in range(len(results['labels'])):
        label = results['labels'][i]
        score = results['scores'][i]
        print(f"{label}: {score:.4f}")

except Exception as e:
    print(f"\n--- AN ERROR OCCURRED ---")
    print("The process failed. This is likely a network issue or a problem with the installation.")
    print(f"Error details: {e}")