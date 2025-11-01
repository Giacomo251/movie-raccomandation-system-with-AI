# src/data_loader.py
import pandas as pd

def load_movies(data):
    df = pd.read_csv("data/imdb_movies.csv")
    # Assumiamo colonne tipiche: movieId/title/genres/year/rating/popularity
    # Adatta questa funzione al tuo dataset
    return df