# Importer les bibliothèques
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Générer des données simples
X = np.linspace(-10, 10, 100).astype(np.float32)  # 100 points entre -10 et 10
y = 2 * X + 5 + np.random.normal(0, 2, size=X.shape)  # y = 2X + 5 + un bruit gaussien

# Séparer les données en train et test
N = int(len(X) * 0.8)  # 80% pour l'entraînement
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], y[N:]

# Reshape pour s'assurer que Keras accepte les dimensions
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Définir le modèle simple
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# Compiler le modèle
model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))

# Entraîner le modèle
model.fit(X_train, y_train, epochs=100, verbose=0)

# Faire des prédictions
y_preds = model.predict(X_test)

# Afficher les résultats
plt.figure(figsize=(6, 5))
plt.scatter(X_train, y_train, c="blue", label="Données d'entraînement")
plt.scatter(X_test, y_test, c="green", label="Données de test")
plt.plot(X_test, y_preds, c="red", linewidth=2, label="Prédictions du modèle")
plt.legend()
plt.title("Régression linéaire avec Keras")
plt.xlabel("X")
plt.ylabel("y")
plt.grid()
plt.show()
