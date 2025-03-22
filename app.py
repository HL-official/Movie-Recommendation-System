from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import ast

app = Flask(__name__)

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

# Function to convert list to a maximum of 5 elements
def convert5(obj):
    return obj[:5]

movies['name'] = movies['name'].apply(convert5)
movies['soup'] = movies['description'] + movies['genres'] + movies['name']

# Combine the soup into a single string and convert to lowercase
new_df = movies[['id', 'title', 'type', 'soup']]
new_df['soup'] = new_df['soup'].apply(lambda x: " ".join(x))
new_df['soup'] = new_df['soup'].apply(lambda x: x.lower())

# Define a TF-IDF Vectorizer Object
tfidf = TfidfVectorizer(stop_words='english')

# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(new_df['soup'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()

# Drop duplicates based on 'title'
new_df.drop_duplicates(subset='title', keep='first', inplace=True)

# Recreate the indices, tfidf_matrix, and cosine_sim
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()
tfidf_matrix = tfidf.fit_transform(new_df['soup'])
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(input_text, by_title=True):
    if by_title:
        if input_text in indices:
            idx = indices[input_text]
            sim_scores = cosine_sim[idx]
        else:
            return ["Movie title not found in the database."]
    else:
        input_soup = input_text.lower()
        input_tfidf = tfidf.transform([input_soup])
        sim_scores = linear_kernel(input_tfidf, tfidf_matrix).flatten()

    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return new_df['title'].iloc[movie_indices].tolist()

# Function to get autocomplete options
def get_autocomplete_options(query):
    query = query.lower()
    num_options = 5
    options = new_df[new_df['title'].str.lower().str.startswith(query)]['title']
    return options.head(num_options).tolist()

@app.route('/recommend', methods=['GET'])
def recommend():
    title = request.args.get('title')
    recommendations = get_recommendations(title, by_title=True)
    return jsonify(recommendations)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query')
    options = get_autocomplete_options(query)
    return jsonify(options)

@app.route('/webhook', methods=['POST'])
def webhook():
    req = request.get_json(silent=True, force=True)
    intent_name = req['queryResult']['intent']['displayName']

    if intent_name == "Get Recommendations":
        movie_title = req['queryResult']['parameters']['movie']
        recommendations = get_recommendations(movie_title, by_title=True)
        response = {"fulfillmentText": "Here are some recommendations: " + ", ".join(recommendations)}
    elif intent_name == "Autocomplete":
        query = req['queryResult']['parameters']['query']
        options = get_autocomplete_options(query)
        response = {"fulfillmentText": "Autocomplete options: " + ", ".join(options)}
    else:
        response = {"fulfillmentText": "Sorry, I didn't understand that."}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
