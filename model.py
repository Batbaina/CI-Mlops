# Importation des modules nécessaires
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Désactiver CUDA et forcer l'utilisation du CPU uniquement
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Définition des fonctions de métriques
def mae(y_true, y_pred):
    """Calcule l'erreur absolue moyenne entre y_true et y_pred."""
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy()

def mse(y_true, y_pred):
    """Calcule l'erreur quadratique moyenne entre y_true et y_pred."""
    return tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()

# Création et affichage de la version de TensorFlow
print(f"TensorFlow Version: {tf.__version__}")

# Génération des données
X = np.arange(-100, 100, 4)
y = np.arange(-90, 110, 4)

# Séparation en données d'entraînement et de test
N = 25
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], y[N:]

# Reshape pour s'assurer que les données sont en format 2D
X_train, X_test = X_train.reshape(-1, 1), X_test.reshape(-1, 1)

# Définition du modèle
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),  # Entrée explicite
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compilation du modèle
model.compile(loss="mae",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["mae"])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=50, verbose=1)

# Prédictions
y_preds = model.predict(X_test)

# Calcul des métriques
mae_value = round(mae(y_test, y_preds), 2)
mse_value = round(mse(y_test, y_preds), 2)
print(f"Mean Absolute Error: {mae_value}, Mean Squared Error: {mse_value}")

# Enregistrement des métriques
with open("metrics.txt", "w") as f:
    f.write(f"MAE: {mae_value}, MSE: {mse_value}")

# Affichage des prédictions
plt.figure(figsize=(6, 5))
plt.scatter(X_train, y_train, c="b", label="Training Data")
plt.scatter(X_test, y_test, c="g", label="Testing Data")
plt.scatter(X_test, y_preds, c="r", label="Predictions")
plt.legend()
plt.grid()
plt.title("Model Predictions")
plt.xlabel("X values")
plt.ylabel("Y values")
plt.savefig("model_results.png", dpi=120)
plt.show()
