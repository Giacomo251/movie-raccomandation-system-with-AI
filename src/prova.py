from preprocessing import preprocess
import pandas as pd

df = pd.read_csv("data/imdb_movies.csv")
X_train, X_val, X_test, y_train, y_val, y_test, scaler, mlb = preprocess(df)

print("Shape X_train:", X_train.shape)
print("Example features:", X_train[0][:10])
print("Example label:", y_train[0])
