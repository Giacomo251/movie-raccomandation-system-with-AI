import numpy as np

#activation function and derivatives
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def dsigmoid(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    return 1 - np.tanh(x) ** 2


#dictionary
activations = {
    "sigmoid": (sigmoid, dsigmoid),
    "tanh": (tanh, dtanh)
}


class MLP:
    def __init__(self, layer_sizes, activation="tanh", seed=42):
        """
        layer_sizes: lista del numero di neuroni per strato, es. [input_dim, 16, 8, 1]
        activation: funzione di attivazione ('tanh' o 'sigmoid')
        """
        np.random.seed(seed)
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.act, self.dact = activations[activation]

        # Inizializzazione pesi e bias
        self.W = []  #pesi
        self.b = []  #bias
        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i + 1]
            limit = np.sqrt(6 / (n_in + n_out))                                 #L'obiettivo è scalare l'intervallo di inizializzazione per garantire
                                                                                #che la varianza delle attivazioni e dei gradienti rimanga approssimativamente la stessa attraverso tutti i layer.
                                                                                #Questo aiuta a prevenire problemi come la scomparsa o 
                                                                                #l'esplosione dei gradienti (vanishing/exploding gradient problem) durante la retropropagazione, specialmente nelle reti profonde.
            self.W.append(np.random.uniform(-limit, limit, (n_out, n_in)))
            self.b.append(np.zeros((n_out, 1)))

    # --- Propagazione in avanti ---
    def forward(self, X):
        """
        Calcola l’output della rete per un batch X.
        Salva i valori intermedi per la backpropagation.
        -Each neuron computes z = W x + b
        -Apply the activation function A = f(z)
        -The result becomes the input for the next layer
        """
        A = X.T  # (n_input, n_samples)
        self.z = []
        self.a = [A]
        for W, b in zip(self.W, self.b):
            z = W.dot(A) + b
            A = self.act(z)
            self.z.append(z)
            self.a.append(A)
        return A.T  # (n_samples, n_output)

    # --- Calcolo della loss (errore medio) ---
    def compute_loss(self, y_pred, y_true):
        eps = 1e-8
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) #Binary Cross Entropy (BCE), the classical function

    # --- Backpropagation ---
    def backward(self, X, y_true, lr=0.01):
        
        #update the weight for reduce the loss
        
        m = X.shape[0]
        y_true = y_true.reshape(1, -1)
        
        A_last = self.a[-1]
        delta = A_last - y_true  # derivata del BCE per sigmoid output

        gradsW = []
        gradsb = []

        for i in reversed(range(len(self.W))):
            A_prev = self.a[i]
            dW = (delta.dot(A_prev.T)) / m
            db = np.mean(delta, axis=1, keepdims=True)
            gradsW.insert(0, dW)
            gradsb.insert(0, db)

            if i > 0:
                z_prev = self.z[i - 1]
                delta = self.W[i].T.dot(delta) * self.dact(z_prev)

        #update weight
        for i in range(len(self.W)):
            self.W[i] -= lr * gradsW[i]
            self.b[i] -= lr * gradsb[i]

    #Previsione (probabilità) from 0 to 1
    def predict_proba(self, X):
        return self.forward(X).ravel()

    #Previsione binaria (0/1) 0 or 1
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)



