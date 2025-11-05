# src/data_loader.py
import pandas as pd

def load_movies(data):
    df = pd.read_csv("data/imdb_movies.csv")
    return df