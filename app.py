import streamlit as st
from src.tweet_nlp import convert_thought_to_genres
from src.recommend_engine import recommend_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Thought to Title Recommender",
    page_icon="🎬",
    layout="centered"
)

# --- Application UI ---
st.title("🎬 Thought to Title Recommender")
st.write("Describe the movie you're in the mood for, and the engine will find a title for you.")

# --- User Input ---
thought_text = st.text_area(
    "Enter your movie thought here:",
    placeholder="e.g., 'I want to see something with spaceships, laser battles, and a lone hero fighting a galactic empire.'"
)

# --- Main Logic ---
if st.button("Find My Movie"):
    if thought_text:
        with st.spinner("Analyzing your thought..."):
            # 1. Call the NLP function to get genres
            predicted_genres = convert_thought_to_genres(thought_text)

        if predicted_genres:
            st.success(f"**Detected Genres:** {', '.join(predicted_genres)}")
            
            with st.spinner("Finding the perfect movie..."):
                # 2. Call the recommender function with the genres
                recommended_title = recommend_engine(predicted_genres)
            
            st.subheader("Recommended Title:")
            st.markdown(f"## **{recommended_title}**")
        else:
            st.error("Sorry, I couldn't determine the genres from that. Please try rephrasing your thought.")
    else:
        st.warning("Please enter a thought first!")