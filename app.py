import streamlit as st
from src.tweet_nlp import convert_thought_to_genres
from src.recommend_engine import recommend_engine

# --- Page Configuration ---
st.set_page_config(
    page_title="Thought to Title Recommender",
    page_icon="🎬",
    layout="centered"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .success-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .warning-box {
        padding: 10px;
        border-radius: 5px;
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# --- Application Header ---
st.title("🎬 Thought to Title Recommender")
st.markdown("<p class='big-font'>Turn your movie ideas into perfect recommendations!</p>", unsafe_allow_html=True)
st.write("Describe the movie you're in the mood for, and our AI will find the perfect title for you.")

# --- User Input Section ---
st.markdown("### 🎭 Describe Your Movie Mood")
thought_text = st.text_area(
    "Enter your movie thought here:",
    placeholder="e.g., 'I want to see something with spaceships, laser battles, and a lone hero fighting a galactic empire.'",
    height=120
)

# --- Recommendation Controls ---
col1, col2 = st.columns([1, 3])
with col1:
    top_k = st.slider("Number of recommendations:", 1, 10, 3)
with col2:
    strict_match = st.checkbox("Strict genre matching", value=True, 
                              help="Only recommend movies that contain at least one of your detected genres")

# --- Main Recommendation Logic ---
if st.button("🔍 Find My Movie", type="primary", use_container_width=True):
    if thought_text.strip():
        with st.spinner("🧠 Analyzing your thought..."):
            try:
                # 1. Extract genres using NLP
                predicted_genres = convert_thought_to_genres(thought_text)
                
                if predicted_genres and isinstance(predicted_genres, list) and len(predicted_genres) > 0:
                    # Display detected genres
                    st.markdown(
                        f"<div class='success-box'><strong>🎯 Detected Genres:</strong> {', '.join(predicted_genres)}</div>", 
                        unsafe_allow_html=True
                    )
                    
                    with st.spinner("🍿 Finding your perfect movie..."):
                        try:
                            # 2. Get recommendations
                            recommended_titles = recommend_engine(
                                predicted_genres, 
                                top_k=top_k, 
                                strict_genre_match=strict_match
                            )
                            
                            if isinstance(recommended_titles, list) and len(recommended_titles) > 0:
                                st.subheader("🏆 Your Movie Recommendations:")
                                
                                # Display recommendations in an attractive format
                                for i, title in enumerate(recommended_titles, 1):
                                    st.markdown(f"**{i}.** {title}")
                                    
                            else:
                                st.markdown(
                                    "<div class='error-box'>❌ Sorry, no movies found matching your criteria.</div>", 
                                    unsafe_allow_html=True
                                )
                                
                        except Exception as e:
                            st.markdown(
                                f"<div class='error-box'>❌ Error getting recommendations: {str(e)}</div>", 
                                unsafe_allow_html=True
                            )
                            
                else:
                    st.markdown(
                        "<div class='error-box'>❌ Sorry, I couldn't determine the genres from that. Please try rephrasing your thought.</div>", 
                        unsafe_allow_html=True
                    )
                    st.info("💡 Tip: Try being more specific about genres like 'action', 'comedy', 'horror', 'sci-fi', etc.")
                    
            except Exception as e:
                st.markdown(
                    f"<div class='error-box'>❌ Error analyzing your thought: {str(e)}</div>", 
                    unsafe_allow_html=True
                )
    else:
        st.markdown(
            "<div class='warning-box'>⚠️ Please enter a thought first!</div>", 
            unsafe_allow_html=True
        )

# --- Example Section ---
with st.expander("💡 Need inspiration? Try these examples:"):
    st.write("**Example 1:**")
    st.code("I'm looking for a scary movie with lots of suspense and unexpected twists.")
    
    st.write("**Example 2:**")
    st.code("I want to watch something with superheroes saving the world from aliens.")
    
    st.write("**Example 3:**")
    st.code("Looking for a romantic comedy about second chances and finding love in unexpected places.")

# --- Footer ---
st.markdown("---")
st.caption("✨ Powered by AI • Made with Streamlit")