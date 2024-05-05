import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error


"""
This module provides various utility functions for analyzing and evaluating trained neural network models.

Functions:
- model_loss(fitted_model): Print the loss metrics of the fitted model.
- plot_model_loss(fitted_model): Plot the training and validation loss curves of the fitted model.
- make_prediction(fitted_model, input_row_df): Make predictions using the fitted model.
- save_model(fitted_model, file_name): Save the fitted model to a file.
- load_model(file_name): Load a saved model from a file.
- plot_model_accuracy(fitted_model): Plot the training and validation accuracy curves of the fitted model.
- print_classification_report(fitted_model, y_test, X_test): Print the classification report based on model predictions.
- plot_predictions_scatter(fitted_model, y_test, X_test): Plot the scatter plot of true values vs predicted values.
- print_errors(fitted_model, y_test, scaled_X_test): Print the root mean squared error (RMSE) and mean absolute error (MAE).
- plot_residual_error(fitted_model, y_test, scaled_X_test): Plot the residual errors of the model predictions.
- get_classes_(data, target_column): Get the unique class labels from the target column of the dataset.
- get_last_loss(fitted_model): Get the last loss value from the training history of the model.
- get_features_(data): Get the list of feature column names from the dataset.
- get_n_layers(fitted_model): Get the number of layers in the model.
- get_n_outputs_(fitted_model): Get the number of output units in the last layer of the model.
- get_out_activation_(fitted_model): Get the activation function of the output layer of the model.
- get_attributes(fitted_model, data, target_column): Print various attributes of the model and dataset.

All functions accept the fitted_model, which is an instance of the trained neural network model,
along with other optional parameters such as data, target_column, y_test, X_test, and scaled_X_test.

Note: Some functions may raise errors or return None values if certain information cannot be retrieved or if there are issues with the input data or model.

"""

def model_loss(fitted_model):
    history = fitted_model.history
    print("Existing metrics:")
    if 'loss' in history.history:
        print("  Loss:", history.history['loss'])
    if 'val_loss' in history.history:
        print("  Val_Loss:", history.history['val_loss'])
    if 'accuracy' in history.history:
        print("  Accuracy:", history.history['accuracy'])
    if 'val_accuracy' in history.history:
        print("  Val_Accuracy:", history.history['val_accuracy'])

def plot_model_loss(fitted_model):
    history = fitted_model.history
    losses = pd.DataFrame({
        'loss': history.history['loss'],
        'val_loss': history.history['val_loss']
    })

    # Check if the losses DataFrame is not empty
    if losses.empty:
        print("Loss data is empty. Unable to plot.")
        return
    
    epochs = range(1, len(losses) + 1)
    
    # Debugging: Print the range of epochs
    print("Epochs:", epochs)

    epochs = range(1, len(losses) + 1)
    
    plt.plot(epochs, losses['loss'], label='Training loss')
    plt.plot(epochs, losses['val_loss'], label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def make_prediction(fitted_model,input_row_df):
     
     prediction = fitted_model.predict(input_row_df)
     return prediction

def save_model (fitted_model,file_name):

    fitted_model.save(file_name)
    print(f"Original Keras model saved as {file_name}")

def load_model(file_name):
    from tensorflow.keras.models import load_model
    loaded_model = load_model(file_name)
    print(f"Model loaded from {file_name}")
    return loaded_model

def plot_model_accuracy(fitted_model):
    history = fitted_model.history
    
    
    accuracies = pd.DataFrame({
        'accuracy': history.history['accuracy'],
        'val_accuracy': history.history['val_accuracy']
    })
    
    # Check if the accuracies DataFrame is not empty
    if accuracies.empty:
        print("Accuracy data is empty. Unable to plot.")
        return
    
    epochs = range(1, len(accuracies) + 1)
    
    plt.plot(epochs, accuracies['accuracy'], label='Training accuracy')
    plt.plot(epochs, accuracies['val_accuracy'], label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

def print_classification_report(fitted_model, y_test, X_test):
    # Get predictions on the test data
    y_true = y_test
    raw_predictions = fitted_model.predict(X_test)
    y_pred = np.argmax(raw_predictions, axis=1)
    
    # Generate classification report
    report = classification_report(y_true, y_pred)
    
    # Print the classification report
    print("Classification Report:")
    print(report)

def plot_predictions_scatter(fitted_model, y_test, X_test):
    
    y_true = y_test
    y_pred = fitted_model.predict(X_test).flatten()
    

    plt.scatter(y_true, y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True Values vs Predictions')
    plt.show()

def print_errors(fitted_model, y_test, scaled_X_test):
    
    # Making predictions
    
    y_pred = fitted_model.predict(scaled_X_test)

    # Calculating errors
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    # Printing errors
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)

    return rmse, mae

def plot_residual_error(fitted_model, y_test, scaled_X_test):
    # Get predictions
    predictions = fitted_model.predict(scaled_X_test)

    # Calculate residuals
    residuals = y_test - predictions.flatten()

    # Plot residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(predictions, residuals, alpha=0.5)
    plt.title('Residual Plot')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.grid(True)
    plt.show()

def get_classes_(data, target_column):
    # Extract unique class labels from the target column
    class_labels = sorted(data[target_column].unique())
    
    # Return the unique class labels
    return class_labels

def get_last_loss(fitted_model):
    # Last Loss 
    history = fitted_model.history
    if history:
        if 'loss' in history.history:
            last_loss = history.history['loss'][-1] 
            return last_loss, None  # Return last_loss and no error message
        else:
            return None, "Loss information not found in the training history."
    else:
        return None, "Training history not available. Unable to retrieve last loss."

def get_features_(data):
    # Extract feature column names
    features_ = data.columns.tolist()
    return features_

def get_n_layers(fitted_model):
    # Check if the model has layers
    if fitted_model.layers:
        # Get the number of layers
        n_layers = len(fitted_model.layers)
        return n_layers, None
    else:
        return None, "Model does not contain any layers."  

def get_n_outputs_(fitted_model):
    if fitted_model:
        n_outputs_ = fitted_model.output_shape[-1]
        return n_outputs_
    else:
        print("Model not found. Unable to retrieve number of outputs.")
        return None

def get_out_activation_(fitted_model):
    if fitted_model:
        out_activation_ = fitted_model.layers[-1].activation.__name__
        return out_activation_
    else:
        print("Model not found. Unable to retrieve output activation function.")
        return None

def get_attributes(fitted_model, data, target_column):
    # Classes_
    classes_ = get_classes_(data, target_column)
    print("Unique Classes:", classes_)

    # Get last loss
    last_loss, error_message = get_last_loss(fitted_model)
    if error_message:
        print(error_message)
    else:
        print("Last Loss:", last_loss)
    
    # Get Features
    features_ = get_features_(data)
    print("Features:", features_)
    
    # Number of Layers
    n_layers, error_message = get_n_layers(fitted_model)
    if error_message:
        print(error_message)
    else:
        print("Number of Layers:", n_layers)

    # n_outputs_
    n_outputs_ = get_n_outputs_(fitted_model)
    if n_outputs_:
        print("Number of Outputs:", n_outputs_)
    else:
        print("Unable to retrieve number of outputs.")

    # out_activation_
    out_activation_ = get_out_activation_(fitted_model)
    if out_activation_:
        print("Output Activation Function:", out_activation_)
    else:
        print("Unable to retrieve output activation function.")