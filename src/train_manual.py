import numpy as np
import matplotlib.pyplot as plt
from mlp_manual import MLP
from preprocessing import preprocess
from data_loader import load_movies
from utils import save_weights
from sklearn.metrics import accuracy_score, confusion_matrix


def train(X_train, y_train, X_val, y_val, layer_sizes, epochs=100, lr=0.01, batch_size=32):
    model = MLP(layer_sizes, activation='tanh')     # initialize the neural network
    history = {'loss': [], 'val_loss': []}          # track training & validation loss

    n = X_train.shape[0]

    for epoch in range(epochs):
        # Shuffle data
        idx = np.random.permutation(n)
        X_train = X_train[idx]
        y_train = y_train[idx]

        # Mini-batch training
        for i in range(0, n, batch_size):
            Xb = X_train[i:i + batch_size]
            yb = y_train[i:i + batch_size]

            # Forward pass
            model.forward(Xb)
            # Backpropagation
            model.backward(Xb, yb, lr=lr)

        # Evaluate on train & val
        y_pred_train = model.forward(X_train).ravel()
        y_pred_val = model.forward(X_val).ravel()

        train_loss = model.compute_loss(y_pred_train, y_train)
        val_loss = model.compute_loss(y_pred_val, y_val)

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch + 1}/{epochs} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f}")

    # Plot training progress
    plt.plot(history['loss'], label="Training loss")
    plt.plot(history['val_loss'], label="Validation loss")
    plt.xlabel("Epoca")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Andamento dell'errore durante l'addestramento")
    plt.show()

    return model, history


if __name__ == "__main__":
    # Load dataset
    df = load_movies("data/imdb_movies.csv")

    # Preprocessing
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, mlb, vectorizer, X_combined, df = preprocess(df)

    # MLP architecture
    input_dim = X_train.shape[1]
    layer_sizes = [input_dim, 32, 16, 1]

    # Training parameters
    epochs = 500
    lr = 0.01
    batch_size = 32

    # Train
    model, hist = train(X_train, y_train, X_val, y_val, layer_sizes, epochs, lr, batch_size)

    # Test
    y_pred_test = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred_test)
    cm = confusion_matrix(y_test, y_pred_test)

    print("\n=== RISULTATI FINALI ===")
    print(f"Accuracy sul test: {acc * 100:.2f}%")
    print("Matrice di confusione:")
    print(cm)

    # Save weights
    save_weights(model, "data/weights_movie_recommender.npz")
    print("\nModello salvato in: data/weights_movie_recommender.npz")
