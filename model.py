# %%
import numpy as np
import pandas as pd
import ast

# %%
df_titles = pd.read_csv('titles.csv')

# %%
df_credits = pd.read_csv('credits.csv')


# %%
df_credits = df_credits[['person_id','id','name']]


# %%
df_credits = df_credits.groupby('id')['name'].apply(','.join).reset_index()


# %%
df = pd.merge(df_titles, df_credits, on='id')

# %%
movies = df[['id','title','type','description','genres','name']]

# %%
movies.dropna(inplace=True)
# %%
movies['description'] = movies['description'].apply(lambda x: x.replace(" — "," "))

# %%
movies['name']=movies['name'].apply(lambda x: x.replace(" ",""))


# %%
movies['name']= movies['name'].apply(lambda x: x.replace(","," "))

# %%
movies['description']=movies['description'].apply(lambda x:x.split())
movies['name']=movies['name'].apply(lambda x:x.split())

# %%
movies['description']=movies['description'].apply(lambda x: [i.replace("-","") for i in x])


# %%
movies['genres'] = movies['genres'].apply(ast.literal_eval)

# %%
def convert5(obj):
    L=[]
    counter = 0
    for i in obj:
        if counter !=5:
            L.append(i)
            counter+=1
        else:
            break

    return L

# %%
movies['name']=movies['name'].apply(convert5)

# %%
movies['soup'] = movies['description']+movies['genres']+movies['name']

# %%
new_df=movies[['id','title','type','soup']]

# %%
new_df['soup'] = new_df['soup'].apply(lambda x:" ".join(x))

# %%
new_df['soup'] = new_df['soup'].apply(lambda x:x.lower())

# %%
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# %%
# Define a TF-IDF Vectorizer Object.
tfidf = TfidfVectorizer(stop_words='english')

# %%
# Construct the required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(new_df['soup'])

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Construct a reverse map of indices and movie titles
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()

# %%
# Drop duplicates based on 'title'
new_df.drop_duplicates(subset='title', keep='first', inplace=True)

# Recreate the indices, tfidf_matrix and cosine_sim
indices = pd.Series(new_df.index, index=new_df['title']).drop_duplicates()

tfidf_matrix = tfidf.fit_transform(new_df['soup'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)


# %%

def get_recommendations(title, cosine_sim=cosine_sim):
     # Get the index of the movie that matches the title
    idx = indices[title]

    # Get the pairwise similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Flatten the array
    sim_scores = [(i, score) for i, score in sim_scores if score.size == 1]

    # Sort the movies based on the similarity scores
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