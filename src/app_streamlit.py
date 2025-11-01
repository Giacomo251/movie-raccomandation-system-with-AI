import streamlit as st
import pandas as pd
import numpy as np
from preprocessing import preprocess
from data_loader import load_movies
from mlp_manual import MLP
from utils import load_weights
from sklearn.metrics.pairwise import cosine_similarity


@st.cache_data      #to avoid loading everything at each iteration
def load_data():
    # load the data
    df = load_movies("data/imdb_movies.csv")
    _, _, _, _, _, _, _, _, _, X_combined, df = preprocess(df)
    # Ricostruiamo X completo per la raccomandazione
    
    return df, X_combined


def get_similar_movies(movie_idx, X_features, df, top_k=5):
    # find similar film with similar of cosine (contend based)
    sims = cosine_similarity(X_features[movie_idx].reshape(1, -1), X_features)[0]
    indices = np.argsort(-sims)  # order
    indices = [i for i in indices if i != movie_idx]
    top_indices = indices[:top_k]

    similar_df = df.iloc[top_indices][['names', 'score', 'genre', 'orig_lang', 'country']].copy()
    similar_df['similarità'] = np.round(sims[top_indices], 3)
    return similar_df




    

# Streamlit

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎬 Movie Recommender System")
st.write("Un piccolo sistema di raccomandazione di film basato su rete neurale MLP (content-based).")

# load data
df, X_combined = load_data()


# --- Selezione film ---
movie_titles = df['names'].tolist()
selected_movie = st.selectbox("Scegli un film:", movie_titles)

# Ottieni indice del film scelto
movie_idx = df[df['names'] == selected_movie].index[0]

st.subheader(f"📄 Dettagli di **{selected_movie}**")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Anno:** {df.iloc[movie_idx]['date_x']}")
    st.write(f"**Genere:** {df.iloc[movie_idx]['genre']}")
    st.write(f"**Lingua:** {df.iloc[movie_idx]['orig_lang']}")
with col2:
    st.write(f"**Punteggio:** {df.iloc[movie_idx]['score']}")
    st.write(f"**Paese:** {df.iloc[movie_idx]['country']}")
    st.write(f"**Budget:** {df.iloc[movie_idx]['budget_x']:.0f}")
    st.write(f"**Revenue:** {df.iloc[movie_idx]['revenue']:.0f}")

st.divider()

# --- Raccomandazioni simili ---
st.subheader("🎞️ Film simili consigliati")
similar_df = get_similar_movies(movie_idx, X_combined, df, top_k=5)
st.table(similar_df)


# --- Sezione rete neurale ---
st.divider()
st.subheader("🧠 Valutazione MLP (opzionale)")

use_model = st.checkbox("Usa modello MLP per stimare se il film piacerebbe")
if use_model:
    # usa X_combined (non X_features)
    input_dim = X_combined.shape[1]
    layer_sizes = [input_dim, 32, 16, 1]
    model = MLP(layer_sizes, activation='tanh')

    # Carica pesi salvati
    weights_path = "data/weights_movie_recommender.npz"
    try:
        model = load_weights(model, weights_path)
        # Assicurati che movie_idx sia valido per X_combined
        if movie_idx < 0 or movie_idx >= X_combined.shape[0]:
            st.error("Indice film non valido per X_combined — assicurati che il DataFrame e le feature siano allineati.")
        else:
            x_single = X_combined[movie_idx].reshape(1, -1)
            prob = model.predict_proba(x_single)[0]
            st.write(f"**Probabilità che piaccia:** {prob * 100:.2f}%")
    except Exception as e:
        st.error("⚠️ Errore nel caricamento dei pesi. Assicurati che il file esista e che l'architettura corrisponda.")
        st.code(str(e))
