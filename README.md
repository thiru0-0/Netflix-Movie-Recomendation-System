# 🎬 Movie Recommendation System

## Project Title
**Netflix Movie Recommendation System** - An intelligent web-based application that provides personalized movie recommendations using content-based filtering and curated recommendation algorithms.

---

## Problem Statement
With thousands of movies available on streaming platforms like Netflix, users often struggle with decision fatigue when choosing what to watch. Finding similar or related movies requires manual browsing through catalogs, which is time-consuming and inefficient. There's a need for an automated system that can quickly recommend movies based on user input, helping users discover new content that matches their preferences and interests.

---

## Project Objective
The objective of this project is to develop an intuitive, user-friendly web application that:
- Allows users to search for movies by title
- Provides relevant movie recommendations in real-time
- Uses intelligent algorithms to match user preferences with similar movies
- Handles user input variations and typos gracefully
- Delivers fast and accurate recommendations from the Netflix catalog

---

## Project Features

### ✨ Core Features
1. **Smart Movie Search**
   - Fuzzy title matching to handle typos and partial matches
   - Autocomplete-like functionality with confidence scoring
   - Displays match confidence to users

2. **Dual Recommendation Engine**
   - **Curated Recommendations**: Uses pre-computed or hand-curated movie IDs for high-precision matches
   - **Content-Based Fallback**: TF-IDF algorithm analyzes movie metadata (title, genres) to find similar movies when curated data isn't available

3. **Interactive Web Interface**
   - Built with Streamlit for a clean, responsive user experience
   - Sidebar controls for easy configuration
   - Real-time recommendation display

4. **Flexible Data Management**
   - Customizable CSV data source
   - Support for movie metadata including title, genre, and recommendation IDs
   - Efficient data caching for improved performance

5. **Scalable Architecture**
   - Handles large datasets efficiently
   - Optional scikit-learn integration for advanced TF-IDF similarity calculations
   - Configurable number of recommendations (default: top 10 movies)

### 🔧 Technical Features
- **Fuzzy Matching**: Uses Python's `difflib` for tolerance of spelling variations
- **TF-IDF Vectorization**: Natural language processing to understand movie relationships
- **Cosine Similarity**: Mathematical algorithm to find the most similar movies
- **Data Caching**: Streamlit caching mechanism for fast load times

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Create a virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Or Windows CMD:
.\.venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt
```

### Running the Application
```bash
# Option 1: Run with Streamlit (recommended)
streamlit run app.py

# Option 2: Run with Python
python main.py
```

The application will open in your default browser at `http://localhost:8501`

---

## Usage

1. **Enter a Movie Title**: Type a movie name in the search field (e.g., "The Matrix")
2. **View Recommendations**: The system displays up to 10 similar movie recommendations
3. **Confidence Score**: Check the confidence percentage to see how well the input matched a movie in the database

---

## Demo Video
*[Add your demo video link here - e.g., YouTube link or recorded GIF]*

---

## Dataset
The project uses the Netflix movie dataset (`netflix_data.csv`) with the following required columns:
- `N_id`: Unique identifier for each movie
- `Title`: Movie title
- `Recommendations`: Comma-separated list of related movie IDs
- `Main Genre`: Primary genre (optional, improves TF-IDF)
- `Sub Genres`: Secondary genres (optional, improves TF-IDF)

---

## How It Works

### Recommendation Algorithm Flow
1. **User Input**: User enters a movie title
2. **Fuzzy Matching**: System finds the closest matching title in the database
3. **Recommendation Path**:
   - If curated recommendations exist → Return mapped movie titles
   - If not → Use TF-IDF + Cosine Similarity to find similar movies
4. **Output**: Display top-K (default: 10) unique recommendations

---

## Technologies Used
- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning for TF-IDF and similarity calculations
- **Python**: Core programming language

---

## Limitations & Future Enhancements

### Current Limitations
- TF-IDF uses only text metadata (doesn't consider ratings, watch history, user embeddings)
- Quality depends on dataset consistency and completeness
- Fuzzy matching may pick unintended titles with ambiguous input

### Future Enhancements
- Integration with user ratings and watch history
- Collaborative filtering for personalized recommendations
- Deep learning embeddings for better movie similarity
- Multi-language support
- Genre-specific filtering options
- Real-time performance monitoring

---

## Project Structure
```
Movie Recomendation System/
├── app.py                          # Main Streamlit application
├── main.py                         # Python entry point
├── netflix_data.csv               # Movie dataset
├── NETFLIX MOVIE RECOMENDATIONS.ipynb  # Jupyter notebook (analysis)
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## Author
[Your Name]

## License
[Add your license here - e.g., MIT License]

---

## Contact
For questions or suggestions, please feel free to reach out!

## Project structure
- `app.py`: Streamlit UI, data loading, fuzzy match, recommendations, TF-IDF fallback.
- `main.py`: Optional Python entry point that launches Streamlit.
- `requirements.txt`: Dependencies.
- `netflix_data.csv`: Your dataset.

## Next steps (ideas)
- Add hybrid scoring: combine curated IDs with TF-IDF scores when both exist.
- Add filters (year, maturity rating, language) and explain-why snippets.
- Persist an index (e.g., precomputed TF-IDF) to speed cold starts.
- Replace TF-IDF with embeddings (e.g., Sentence-Transformers) for semantic similarity.
