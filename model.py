# Import modules and packages
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Functions and procedures
def plot_predictions(train_data, train_labels, test_data, test_labels, predictions):
    """
    Plots training data, test data, and compares predictions.
    """
    plt.figure(figsize=(6, 5))
    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", label="Training data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", label="Testing data")
    # Plot predictions in red
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    # Show the legend
    plt.legend(shadow=True)  # FIXED: shadow=True (boolean)
    # Set grids
    plt.grid(which='major', c='#cccccc', linestyle='--', alpha=0.5)
    # Labels
    plt.title('Model Results', family='Arial', fontsize=14)
    plt.xlabel('X axis values', family='Arial', fontsize=11)
    plt.ylabel('Y axis values', family='Arial', fontsize=11)
    # Save the figure
    plt.savefig('model_results.png', dpi=120)

def mae(y_true, y_pred):
    """
    Calculates mean absolute error (MAE).
    """
    return tf.keras.losses.MAE(y_true, y_pred).numpy()  # FIXED: Using correct MAE function

def mse(y_true, y_pred):
    """
    Calculates mean squared error (MSE).
    """
    return tf.keras.losses.MSE(y_true, y_pred).numpy()  # FIXED: Using correct MSE function

# Check TensorFlow version
print(f"TensorFlow Version: {tf.__version__}")

# Create features (input data)
X = np.arange(-100, 100, 4).astype(np.float32)  # Convert to float for TF compatibility

# Create labels (output data)
y = np.arange(-90, 110, 4).astype(np.float32)  # Convert to float for TF compatibility

# Split data into train and test sets
N = 25
X_train = X[:N]  # First 25 examples (training data)
y_train = y[:N]

X_test = X[N:]  # Remaining examples (test data)
y_test = y[N:]

# FIXED: Reshape input data to ensure correct shape (N, 1)
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Set random seed for reproducibility
tf.random.set_seed(1989)

# Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)),  # Ensure correct input shape
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(loss=tf.keras.losses.MAE,  # Mean Absolute Error
              optimizer=tf.keras.optimizers.SGD(),
              metrics=['mae'])

# Train the model
model.fit(X_train, y_train, epochs=20)

# Make predictions
y_preds = model.predict(X_test)

# Plot predictions
plot_predictions(train_data=X_train, train_labels=y_train,
                 test_data=X_test, test_labels=y_test, predictions=y_preds)

# Calculate model evaluation metrics
mae_value = round(mae(y_test, y_preds), 2)  # FIXED: Proper function usage
mse_value = round(mse(y_test, y_preds), 2)  # FIXED: Proper function usage

# Display results
print(f'\nMean Absolute Error = {mae_value}, Mean Squared Error = {mse_value}.')

# Write metrics to file
with open('metrics.txt', 'w') as outfile:
    outfile.write(f'\nMean Absolute Error = {mae_value}, Mean Squared Error = {mse_value}.')
