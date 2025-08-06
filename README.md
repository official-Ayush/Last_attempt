# 🎬 Thought-to-Title NLP Recommender System

An intelligent, end-to-end web application that recommends a movie title by interpreting a user's thoughts and moods expressed in natural language.

---

## 🚀 Core Features

- **Natural Language Understanding**  
  Interprets free-form text to understand a user's mood and specific requests using a hybrid NLP pipeline.

- **Intelligent Recommendation**  
  Utilizes a content-based filtering model with Cosine Similarity to find the most relevant movie from a database.

- **Interactive Web App**  
  A clean, user-friendly interface built with Streamlit allows for real-time interaction and recommendations.

- **Modular Architecture**  
  Built with a clean, decoupled structure that separates the NLP logic, recommendation engine, and front-end application.

---

## 🧠 How It Works

The project follows a two-stage pipeline to go from a user's thought to a final movie title recommendation:


### 📘 NLP Pipeline (`tweet_nlp.py`)

- A **"keyword-first"** approach is used. The system first scans the user's input for explicit keywords (e.g., `"space"`, `"crime"`, `"love"`).
- If no keywords are found, it falls back to a **Hugging Face Transformers model** (`distilbert-base-uncased-finetuned-sst-2-english`) to perform **sentiment analysis**.
- The detected **mood** (POSITIVE/NEGATIVE) is mapped to a pre-defined list of genres  
  _(e.g., NEGATIVE → Drama, Thriller)_.
- The final output is a clean **list of predicted genres**.

### 🎯 Recommender Engine (`recommend_engine.py`)

- Loads a pre-trained model (`genre_to_title_model.pkl`) containing:
  - A **CountVectorizer**
  - A **movie-genre matrix**
- Converts the predicted genres into a **numerical vector**.
- Computes **Cosine Similarity** between the input and all movie vectors.
- Returns the **movie title** with the highest similarity score.

---

## 🧰 Tech Stack

- **Application Framework**: Streamlit  
- **Machine Learning**: Scikit-learn  
- **NLP**: Hugging Face Transformers  
- **Core Libraries**: Pandas, NumPy, Pickle  

---

## 📁 Project Structure

Boring_Project1/
│
├── app.py # The main Streamlit application script
├── train_recommender.py # Script to train and save the recommender model
├── README.md # Project documentation
│
├── data/
│ └── movies.csv # Dataset with movie titles and genres
│
├── models/
│ └── genre_to_title_model.pkl # Saved recommender model
│
└── src/
├── init.py
├── recommend_engine.py # Title recommendation logic
└── tweet_nlp.py # Thought-to-genre NLP logic


---

## ⚙️ Setup and Installation

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd Boring_Project1
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate

Create a requirements.txt file:

streamlit
pandas
numpy
scikit-learn
torch
transformers

Then run:

pip install -r requirements.txt

4. Train the Recommender Model

Before running the app, generate the genre_to_title_model.pkl file:

python train_recommender.py

🚀 How to Run the App

Once setup is complete:

streamlit run app.py

A new tab will open in your browser with the running application.

Enjoy the movie magic! 🎥✨


---
