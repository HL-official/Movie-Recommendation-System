import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load datasets
df_titles = pd.read_csv('titles.csv')
df_credits = pd.read_csv('credits.csv')

# Process credits
df_credits = df_credits[['person_id', 'id', 'name']]
df_credits = df_credits.groupby('id')['name'].apply(','.join).reset_index()

# Merge datasets
df = pd.merge(df_titles, df_credits, on='id')

# Select relevant columns and drop missing values
movies = df[['id', 'title', 'type', 'description', 'genres', 'name']]
movies.dropna(inplace=True)

# Clean and preprocess data
movies['description'] = movies['description'].apply(lambda x: x.replace(" — ", " "))
movies['name'] = movies['name'].apply(lambda x: x.replace(" ", ""))
movies['name'] = movies['name'].apply(lambda x: x.replace(",", " "))
movies['description'] = movies['description'].apply(lambda x: x.split())
movies['name'] = movies['name'].apply(lambda x: x.split())
movies['description'] = movies['description'].apply(lambda x: [i.replace("-", "") for i in x])
movies['genres'] = movies['genres'].apply(ast.literal_eval)

def convert5(obj):
    L = []
    counter = 0
    for i in obj:
        if counter != 5:
            L.append(i)
            counter += 1
        else:
            break
    return L

movies['name'] = movies['name'].apply(convert5)
movies['soup'] = movies['description'] + movies['genres'] + movies['name']

# Combine the soup into a single string and convert to lowercase
new_df = movies[['id', 'title', 'type', 'soup']]
new_df['soup'] = new_df['soup'].apply(lambda x: " ".join(x))
new_df['soup'] = new_df['soup'].apply(lambda x: x.lower())

# Define a TF-IDF Vectorizer Object
tfidf = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=3, ngram_range=(1, 2))

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(new_df['soup'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()

# Drop duplicates based on 'title'
new_df.drop_duplicates(subset='title', keep='first', inplace=True)

# Recreate the indices, tfidf_matrix and cosine_sim
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()
tfidf_matrix = tfidf.fit_transform(new_df['soup'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

def get_recommendations(input_text, by_title=True, cosine_sim=cosine_sim):
    if by_title:
        # Get the index of the movie that matches the title
        idx = indices[input_text]
        
        # Get the pairwise similarity scores of all movies with that movie
        sim_scores = cosine_sim[idx]
    else:
        # Use the provided description directly as soup
        input_soup = input_text.lower()
        input_tfidf = tfidf.transform([input_soup])
        sim_scores = linear_kernel(input_tfidf, tfidf_matrix).flatten()
    
    # Sort the movies based on the similarity scores
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return new_df['title'].iloc[movie_indices]

def get_autocomplete_options(query):
    query = query.lower()  # Convert the query to lowercase
    num_options = 5  # Number of autocomplete options to return

    # Filter the DataFrame to rows where the movie title starts with the query string
    options = new_df[new_df['title'].str.lower().str.startswith(query)]['title']

    # Convert the filtered DataFrame to a list and return it
    return options.head(num_options).tolist()

# Example: Get recommendations for the movie "Inception"
recommended_movies = get_recommendations("Inception", by_title=True)
print("Recommended movies based on 'Inception':")
print(recommended_movies)

# Example: Get recommendations based on a description (soup)
description = "thriller LeonardoDiCaprio"
recommended_movies_with_soup = get_recommendations(description, by_title=False)
print("Recommended movies based on the provided description:")
print(recommended_movies_with_soup)

# Example: Get autocomplete options for the query "Inc"
autocomplete_options = get_autocomplete_options("Inc")
print("Autocomplete options for 'Inc':")
print(autocomplete_options)