import numpy as np
import pandas as pd #for dataset
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer, OneHotEncoder #for binary
from sklearn.model_selection import train_test_split #for train, validation, test
from sklearn.feature_extraction.text import TfidfVectorizer



#extracts the year from the date column
def extract_year(date_str):
    if pd.isna(date_str):
        return np.nan
    try:
        return int(str(date_str).split('/')[-1])
    except:
        return np.nan





def preprocess(df, test_size=0.3, val_size=0.1, score_threshold=70):
    #makes a copy of the DataFrame, adds a new column year, removes rows that do not have year or score
    df = df.copy()
    df['year'] = df['date_x'].apply(extract_year)
    df = df.dropna(subset=['year', 'score'])

    #numeric values ​​between 0 and 1 are normalized
    numeric_cols = ['year', 'score', 'budget_x', 'revenue']
    num_df = df[numeric_cols].fillna(0)

    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(num_df)

    #take the genre column and create other columns with values ​​0 or 1
    if 'genre' in df.columns:
        genres = df['genre'].fillna('').apply(lambda s: [g.strip() for g in s.split(',') if g.strip() != ''])
        mlb = MultiLabelBinarizer()
        genre_encoded = mlb.fit_transform(genres)
        genre_df = pd.DataFrame(genre_encoded, columns=[f"g_{g}" for g in mlb.classes_])
    else:
        genre_df = pd.DataFrame(index=df.index)
        mlb = None

    #the same of the previus function but with the lang and country
    cat_features = []
    for col in ['orig_lang', 'country']:
        if col in df.columns:
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            cat_encoded = enc.fit_transform(df[[col]].fillna('Unknown'))
            cat_df = pd.DataFrame(cat_encoded, columns=[f"{col}_{c}" for c in enc.categories_[0]])
            cat_features.append(cat_df)
    cat_df_final = pd.concat(cat_features, axis=1) if cat_features else pd.DataFrame(index=df.index)

    #the values are concatenated in the matrix X

    # --- 🔹 TF-IDF sulla trama (overview) (Term Frequency-Inverse Document Frequency)
    vectorizer = TfidfVectorizer(max_features=300, stop_words='english')
    overview_tfidf = vectorizer.fit_transform(df['overview'].fillna(''))

    # --- 🔹 Combina tutto in un’unica matrice ---
    #    Pesi: testo 0.7, genere 0.2, numeriche+categoriche 0.1
    X_combined = np.hstack([
        overview_tfidf.toarray() * 0.6,
        genre_df.values * 0.3,
        np.hstack([X_num, cat_df_final.values]) * 0.1
    ])

    #Y = if score >= 70(score_thre) y=1, else y=0
    y = (df['score'] >= score_threshold).astype(int).values

    #train / validation / test
    X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y, test_size=test_size, random_state=42)
    val_fraction = val_size / test_size
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_fraction, random_state=42)
    
    
    print(f"Training: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
    print(f"Feature totali: {X_combined.shape[1]}")

    # Ritorna anche la matrice completa X_combined per la raccomandazione
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, mlb, vectorizer, X_combined, df
