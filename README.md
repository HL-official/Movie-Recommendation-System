# Movie Recommendation System

A Flask-based movie recommendation system that uses content-based filtering to suggest similar movies based on descriptions, genres, and cast information.

## Features
- Movie recommendations based on title
- Autocomplete suggestions for movie titles
- RESTful API endpoints
- Chatbot integration support

## Setup
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Place your movie datasets:
- `titles.csv`: Contains movie information (can get it from kaggle)
- `credits.csv`: Contains movie credits information (can get it from kaggle)

3. Run the application:
```bash
python app.py
```

## API Endpoints
- `/recommend?title=<movie_title>`: Get movie recommendations
- `/autocomplete?query=<partial_title>`: Get autocomplete suggestions
- `/webhook`: Chatbot integration endpoint
