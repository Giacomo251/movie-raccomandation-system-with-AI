import numpy as np
import os


# ==========================
# SALVATAGGIO PESI
# ==========================
def save_weights(model, path):
    """
    Salva i pesi (W) e bias (b) di un modello MLP in formato .npz.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    weights = {}
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        weights[f"W{i}"] = W
        weights[f"b{i}"] = b

    np.savez(path, **weights)
    print(f"[INFO] Pesi salvati in: {path}")


# ==========================
# CARICAMENTO PESI
# ==========================
def load_weights(model, path):
    """
    Carica i pesi (W) e bias (b) da un file .npz in un modello MLP esistente.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File non trovato: {path}")

    data = np.load(path)
    n_layers = len(model.W)

    for i in range(n_layers):
        W_key = f"W{i}"
        b_key = f"b{i}"

        if W_key in data and b_key in data:
            model.W[i] = data[W_key]
            model.b[i] = data[b_key]
        else:
            raise KeyError(f"Mancano i pesi o bias per lo strato {i}")

    print(f"[INFO] Pesi caricati da: {path}")
    return model


# ==========================
# ALTRE UTILITY OPZIONALI
# ==========================
def set_seed(seed=42):
    """
    Imposta un seed globale per risultati riproducibili.
    """
    np.random.seed(seed)


def print_model_info(model):
    """
    Stampa un riepilogo della struttura del modello MLP.
    """
    print("\n=== STRUTTURA MODELLO ===")
    for i, (W, b) in enumerate(zip(model.W, model.b)):
        print(f"Layer {i + 1}: {W.shape[1]} → {W.shape[0]}  |  Pesi: {W.size}")
    print("==========================\n")
