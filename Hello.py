import streamlit as st
import pandas as pd
import numpy as np
from six import iteritems
from six.moves import xrange


def idx_movie_by_title(title, df):
  try:
    movie_index = df[df["title"] == title].index[0]
    print(f"The index of '{title}' is {movie_index}")
    return movie_index
  except IndexError:
    print(f"'{title}' not found in the DataFrame")
    return -1
  
class BM25(object):

    def __init__(self, corpus):

        self.corpus_size = len(corpus)
        # self.avgdl = sum(float(len(x)) for x in corpus) / self.corpus_size
        self.avgdl = sum(len(str(x)) for x in df.overview) / len(df.overview)
        self.corpus = corpus
        self.tf = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.initialize()

    def initialize(self):

        for document in self.corpus:
            frequencies = {}
            self.doc_len.append(len(document))
            for word in document:
                if word not in frequencies:
                    frequencies[word] = 0
                frequencies[word] += 1
            self.tf.append(frequencies)

            for word, freq in iteritems(frequencies):
                if word not in self.df:
                    self.df[word] = 0
                self.df[word] += 1

        for word, freq in iteritems(self.df):
            self.idf[word] = np.log((self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_score(self, query_document, index, k1=2.5, b=0.85, e=0.2):
        score = 0
        for word in query_document:
            if word not in self.tf[index]:
                continue
            score += (self.idf[word] * self.tf[index][word] * (k1 + 1)
                      / (self.tf[index][word] + k1 * (1 - b + b * self.doc_len[index] / self.avgdl)))
        return score

    def get_bm25_scores(self, query_document):
        scores = []
        for index in xrange(self.corpus_size):
            score = self.get_score(query_document, index)
            scores.append( score)
        return scores

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1) + len(set2) - intersection
    return intersection / union if union > 0 else 0.0


# Load your dataset (replace 'your_dataset.csv' with the actual dataset file)
df = pd.read_csv('preprocessed_data.csv')

delimiter = ','
df['genres'] = df['genres'].apply(lambda x: x.split(delimiter) if isinstance(x, str) else x)
df['overview'] = df['overview'].apply(lambda x: x.split(delimiter) if isinstance(x, str) else x)

df['overview'] = df['overview'].fillna('').apply(lambda x: [] if len(x) == 0 else x)
df['genres'] = df['genres'].fillna('').apply(lambda x: [] if len(x) == 0 else x)


df.info()
# Your BM25 initialization
bm25 = BM25(df.overview)

# Function to get recommendations
def get_recommendations(query_movie_title):
    query_movie_index = idx_movie_by_title(query_movie_title, df)

    # Menghitung nilai jaccard similarity masing-masing dokumen terhadap kueri
    query_genres = set(df.iloc[query_movie_index]["genres"])
    jaccard_scores = []
    for index, row in df.iterrows():
        movie_genres = set(row["genres"])
        jaccard_score = jaccard_similarity(query_genres, movie_genres)
        jaccard_scores.append((index, jaccard_score))

    # Filter movies yang memiliki minimal 1 genre yang sama (Jaccard coefficient > 0)
    min_similar_genre_movies = [index for index, score in jaccard_scores if score > 0]

    # Filter film kueri dari hasil rekomendasi
    min_similar_genre_movies = [index for index in min_similar_genre_movies if index != query_movie_index]

    # Hitung BM25 scores untuk seluruh dokumen film
    bm25_scores = bm25.get_bm25_scores(df.iloc[query_movie_index].overview)

    # Melakukan min-max normalisasi
    min_score = min(bm25_scores)
    max_score = max(bm25_scores)
    normalized_bm25_scores = [(index, (score - min_score) / (max_score - min_score)) for index, score in enumerate(bm25_scores)]

    # Gabungkan nilai jaccard dan bm25
    combined_scores = []
    for index in min_similar_genre_movies:
        jaccard_score = jaccard_scores[index][1]
        normalized_bm25_score = normalized_bm25_scores[index][1]
        combined_score = (jaccard_score * 0.5 + normalized_bm25_score * 0.5)
        combined_scores.append((index, combined_score))

    # Menggolongkan film dari nilai gabungan terbesar hingga terkecil
    combined_scores.sort(key=lambda x: x[1], reverse=True)

    N = 10 
    top_N_movies = combined_scores[:N]

    return top_N_movies

# Streamlit App
st.set_page_config(
        page_title="Movie Recommender System",
        page_icon="📽",
    )

st.write("# Welcome to BM25 Movie Recommender System! 👋")
    

movie_list = df['title'].values

selected_movie = st.selectbox( "Type or select a movie from the dropdown", movie_list,index=None ,placeholder= 'Type here')

if st.button('Show Recommendation'):
    recommended_movie_names = get_recommendations(selected_movie)
    
    row_indices = [tup[0] for tup in recommended_movie_names]

    col = ['title', 'synopsis', 'genres', 'year']
    
    recommended_movies_df = df.iloc[row_indices][col].reset_index(drop= True)
    st.header('Top Movies with Similar Genre and Highest Combined Scores:')
    # st.table(recommended_movies_df)
    st.table(recommended_movies_df.style.set_table_styles(
        [{'selector': 'th',
          'props': [('background', '#424769'), ('color', 'white')]},
         {'selector': 'td',
          'props': [('background', '#2D3250'), ('color', 'white')]}]
    ))
