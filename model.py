# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Functions and procedures
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """Plots training data, test data, and model predictions."""
    plt.figure(figsize=(6, 5))
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend(shadow=True)  # FIX: shadow should be a boolean
    plt.grid(which='major', linestyle='--', alpha=0.5)
    plt.title('Model Results')
    plt.xlabel('X values')
    plt.ylabel('Y values')
    plt.savefig('model_results.png', dpi=120)
    plt.show()

def mae(y_true, y_pred):
    """Calculates mean absolute error."""
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred).numpy()

def mse(y_true, y_pred):
    """Calculates mean squared error."""
    return tf.keras.metrics.mean_squared_error(y_true, y_pred).numpy()

# Check TensorFlow version
print("TensorFlow Version:", tf.__version__)

# Create features and labels
X = np.arange(-100, 100, 4).reshape(-1, 1)  # FIX: Reshaping to 2D
y = np.arange(-90, 110, 4).reshape(-1, 1)  # FIX: Reshaping to 2D

# Split data into train and test sets
N = 25
X_train, y_train = X[:N], y[:N]
X_test, y_test = X[N:], y[N:]

# Set random seed for reproducibility
tf.random.set_seed(1989)

# Create a simple regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),  # FIX: Added activation function
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss='mae',
              optimizer=tf.keras.optimizers.Adam(),  # FIX: Using Adam instead of SGD
              metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20, verbose=1)

# Make predictions
y_preds = model.predict(X_test)

# Plot results
plot_predictions(X_train, y_train, X_test, y_test, y_preds)

# Calculate model metrics
mae_value = round(mae(y_test, y_preds), 2)
mse_value = round(mse(y_test, y_preds), 2)
print(f'\nMean Absolute Error = {mae_value}, Mean Squared Error = {mse_value}.')

# Save metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'Mean Absolute Error = {mae_value}, Mean Squared Error = {mse_value}.')
