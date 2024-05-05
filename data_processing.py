import pandas as pd
import numpy as np
from tensorflow.keras.activations import relu, sigmoid, softmax, tanh






# Class to handle user input for model training
class UserInput:

    """
    A class to handle user input for model training.

    Attributes:
        data: DataFrame or None, the loaded CSV data.
        file_path: str or None, the file path provided by the user.
        ml_type: str or None, the type of machine learning task ('regressor' or 'classifier').
        target_column: str or None, the name of the target column.
        hidden_layer_sizes: tuple of int or float, the sizes of hidden layers in the neural network.
        activation: str, the activation function for hidden layers.
        loss: str, the loss function for model training.
        optimizer: str, the optimizer algorithm for model training.
        batch_size: int, the batch size for training.
        epochs: int, the number of epochs for training.
        monitor: str, the metric to monitor during training.
        patience: int, the patience value for early stopping.
        mode: str, the mode for monitoring (auto/min/max).
        verbose: int, the verbosity level during training.
        multiprocessing: bool, indicating whether to use process-based threading.
        metrics: list of str, the evaluation metrics for the model.
    """

    def __init__ (self):
        """
        Initializes UserInput with default values for attributes.
        """

        self.data = None
        self.file_path = None
        self.ml_type = None
        self.target_column = None
        self.hidden_layer_sizes = (100,)
        self.activation = "relu"
        self.loss = "mse"
        self.optimizer = "adam"
        self.batch_size = 32
        self.epochs = 1
        self.monitor = "val_loss"
        self.patience = self.epochs
        self.mode = "auto"
        self.verbose = 1
        self.multiprocessing = False
        self.metrics = None

    """Execute the sequence of methods to gather user input."""
    def run_all(self):
        # Execute the sequence of methods to gather user input
        self.get_file_path()
        self.ml_type = self.get_ml_type()
        self.target_column = self.get_target_column()
        self.hidden_layer_sizes = self.get_hidden_layers()
        self.activation = self.get_activation()
        self.loss = self.get_loss_function()
        self.optimizer = self.get_optimizer()
        self.batch_size = self.get_batch_size()
        self.epochs = self.get_epochs()
        self.monitor = self.get_monitor()
        self.patience = self.get_patience()
        self.mode = self.get_mode()
        self.verbose = self.get_verbose()
        self.multiprocessing = self.get_multiprocessing()
        self.metrics = self.get_metrics()
    """Prompt the user for a valid file path and load CSV data."""   
    def get_file_path(self):
        # Continuously prompt the user until a valid file path is provided
        while True:
            try:
                file_path = input("Enter the path or name of the CSV file: ")
                print("")
                self.data = pd.read_csv(file_path) # Read CSV data from the provided file path
                self.file_path = file_path # Set the instance variable with the provided file path
                break   # Exit the loop if successful
            except FileNotFoundError:
                print("File not found. Please provide a valid file path.\n") # Inform the user of the error

    """Prompt the user for the type of machine learning: 'regressor' or 'classifier'."""
    def get_ml_type(self):
        # Continuously prompt the user until a valid machine learning type is provided
        while True:
            try:
                # Prompt user for ML type
                ml_type = input("Enter 'regressor' or 'classifier' for the type of machine learning: ").lower()
                if ml_type in ['regressor', 'classifier']:
                    return ml_type # Return the valid ML type if provided
                else:
                    # Inform the user of the error
                    raise ValueError("Invalid input. Please enter either 'regressor' or 'classifier'.\n")
            except ValueError as e:
                print(e) # Print the error message if an exception occurs

    """Prompt the user for the target column and validate its compatibility with the selected ML type."""
    def get_target_column(self):
        # Display available columns in the loaded dataset
        print("\nAvailable columns in the dataset:")
        print(self.data.columns)
        
        while True:
            try:
                target_column = input("Enter the name of the target column: ")
                print("")

                # Check if the entered target column exists in the dataset
                if target_column in self.data.columns:
                    # Check if the target column is of numeric data type
                    if pd.api.types.is_numeric_dtype(self.data[target_column]):
                        # Calculate the proportion of unique values
                        unique_values_count = len(self.data[target_column].unique())
                        total_values_count = len(self.data[target_column])
                        unique_values_proportion = unique_values_count / total_values_count
                        
                        # Determine the threshold for considering the column as categorical
                        categorical_threshold = 0.05  # Adjust as needed
                        
                        if unique_values_proportion <= categorical_threshold:
                            print("Target column appears to be categorical.")
                            return target_column
                        else:
                            print("Target column appears to be continuous. (Regressor is recommended)")

                    else:
                        # Inform the user that the target column is neither categorical nor continuous
                        raise ValueError(f"Unsupported data type for target column '{target_column}'.\n")
                else:
                    # Raise an error if the entered target column is not found in the dataset
                    raise ValueError(f"Column '{target_column}' not found. Please choose a valid target column.\n")
            except ValueError as e:
                print(e)  # Print the error message if an exception occurs
    
    """Prompt the user for custom hidden layer sizes and validate the input"""    
    def get_hidden_layers(self):
        while True:
            try:
                custom_hidden_layers = input("Do you want to choose custom hidden layer sizes? (Y:Yes /N:No) (Default: (100,): ")
                if custom_hidden_layers.lower() == 'y': # Check if user wants custom hidden layer sizes
                    hidden_layers_input = input("Enter the hidden layer sizes separated by commas (e.g., 50,25,-0.25,12): ")
                    if not hidden_layers_input.strip():  # Check if input is empty
                        raise ValueError("Input cannot be empty. Please enter at least one hidden layer size.\n")
                    else:
                        # Convert input string to tuple of floats
                        hidden_layer_sizes = tuple(map(float, hidden_layers_input.split(',')))
                elif custom_hidden_layers.lower() == 'n': # Check if user does not want custom hidden layer sizes
                    print(f"Choosing the default hidden layer sizes: {self.hidden_layer_sizes}\n")
                    hidden_layer_sizes = self.hidden_layer_sizes  # Use default value
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")

                # Validate if hidden layer sizes are provided correctly
                if all(isinstance(size, (int, float)) for size in hidden_layer_sizes): # Check if all elements in hidden_layer_sizes are integers or floats
                    return hidden_layer_sizes
                else:
                    raise ValueError("Invalid input. Please provide valid hidden layer sizes separated by commas.\n")
            except ValueError as e:
                print(e)

    """ Prompt the user for a custom activation function and validate the input."""
    def get_activation(self):
        while True:
            try:
                custom_activation = input("Do you want to choose a custom activation function? (Y:Yes /N:No) (Default: 'relu'): ")
                if custom_activation.lower() == 'y':
                    activation_input = input("Enter the activation function (relu, sigmoid, softmax, tanh): ")
                    if activation_input.lower() not in ['relu','sigmoid', 'softmax', 'tanh']:
                        raise ValueError("Invalid input. Please enter one of these choices: 'sigmoid', 'softmax', 'tanh'.\n")
                    else:
                        # Return the corresponding function object based on user input
                        return activation_input.lower()
                        
                elif custom_activation.lower() == 'n':
                    print(f"Choosing the default activation function: {self.activation}\n")
                    return self.activation  # Use the default activation function
                    
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """Prompt the user for a custom loss function and validate the input."""
    def get_loss_function(self):
        while True:
            try:
                custom_loss = input("Do you want to choose a custom loss function? (Y:Yes /N:No) (Default: 'mse'): ")
                if custom_loss.lower() == 'y':
                    loss_input = input("Enter the loss function (mse, binary_crossentropy, categorical_crossentropy): ")
                    if loss_input.lower() not in ['mse', 'binary_crossentropy', 'categorical_crossentropy']:
                        raise ValueError("Invalid input. Please enter one of these choices: 'binary_crossentropy', 'categorical_crossentropy'.\n")
                    else:
                        return loss_input.lower()  # Return the chosen loss function
                elif custom_loss.lower() == 'n':
                    print(f"Choosing the default loss function: {self.loss}\n")
                    return self.loss  # Use the default loss function
                    
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)    

    """Prompt the user for a custom optimizer and validate the input."""
    def get_optimizer(self):
        while True:
            try:
                custom_optimizer = input("Do you want to choose a custom optimizer? (Y:Yes /N:No) (Default: 'adam'): ")
                if custom_optimizer.lower() == 'y':
                    optimizer_input = input("Enter the optimizer (adam, rmsprop, sgd): ")
                    if optimizer_input.lower() not in ['adam', 'rmsprop', 'sgd']:
                        raise ValueError("Invalid input. Please enter one of these choices: 'adam', 'rmsprop', 'sgd'.\n")
                    else:
                        return optimizer_input.lower()  # Assign the chosen optimizer to self.optimizer
                        
                elif custom_optimizer.lower() == 'n':
                    print(f"Choosing the default optimizer: {self.optimizer}\n")
                    return self.optimizer  # Use the default optimizer
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """Prompt the user for a custom batch size and validate the input."""
    def get_batch_size(self):
        while True:
            try:
                custom_batch_size = input("Do you want to choose a custom batch size? (Y:Yes /N:No) (Default: 32): ")
                if custom_batch_size.lower() == 'y':
                    batch_size_input = input("Enter the batch size (positive number): ")
                    if batch_size_input.strip() == '':
                        raise ValueError("Batch size cannot be empty. Please enter a positive number.\n")
                    else:
                        batch_size = int(batch_size_input)
                        if batch_size <= 0:
                            raise ValueError("Batch size must be a positive number.\n")
                        else:
                            return  batch_size
                elif custom_batch_size.lower() == 'n':
                    print("Choosing the default batch size: (32)\n")
                    return self.batch_size  # Use the default batch size
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """Prompt the user for a custom number of epochs and validate the input."""
    def get_epochs(self):
        while True:
            try:
                custom_epochs = input("Do you want to choose a custom number of epochs? (Y:Yes /N:No) (Default: 1): ")
                if custom_epochs.lower() == 'y':
                    epochs_input = input("Enter the number of epochs (positive number): ")
                    if epochs_input.strip() == '':
                        raise ValueError("Epochs cannot be empty. Please enter a positive number.\n")
                    else:
                        epochs = int(epochs_input)
                        if epochs <= 0:
                            raise ValueError("Epochs must be a positive number.\n")
                        else:
                            return epochs
                elif custom_epochs.lower() == 'n':
                    print("Choosing the default number of epochs: (1)\n")
                    return self.epochs  # Use the default number of epochs
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)
    
    """Prompt the user for a custom metric to monitor and validate the input."""
    def get_monitor(self):
        while True:
            try:
                custom_monitor = input("Do you want to choose a custom metric to monitor? (Y:Yes /N:No) (Default: 'val_loss'): ")
                if custom_monitor.lower() == 'y':
                    monitor_input = input("Enter the metric to monitor (val_loss, accuracy): ")
                    if monitor_input.lower() not in ['val_loss', 'accuracy']:
                        raise ValueError("Invalid input. Please enter one of these choices: 'val_loss', 'accuracy'.\n")
                    else:
                        return monitor_input.lower()
                elif custom_monitor.lower() == 'n':
                    print("Choosing the default metric to monitor: 'val_loss'\n")
                    return self.monitor # Use the default metric to monitor
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """Prompt the user for a custom patience value and validate the input."""
    def get_patience(self):
        while True:
            try:
                custom_patience = input("Do you want to choose a custom patience value? (Y:Yes /N:No) (Default: same as epochs): ")
                if custom_patience.lower() == 'y':
                    patience_input = input("Enter the patience value (positive number): ")
                    if patience_input.strip() == '':
                        raise ValueError("Patience cannot be empty. Please enter a positive number.\n")
                    else:
                        patience = int(patience_input)
                        if patience <= 0:
                            raise ValueError("Patience must be a positive number.\n")
                        else:
                            return patience
                elif custom_patience.lower() == 'n':
                    print(f"Choosing the default patience value: {self.epochs}\n")
                    return self.patience # Use the default patience value
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """ Prompt the user for a custom mode and validate the input."""
    def get_mode(self):
        while True:
            try:
                custom_mode = input("Do you want to choose a custom mode? (Y:Yes /N:No) (Default: 'auto'): ")
                if custom_mode.lower() == 'y':
                    mode_input = input("Enter the mode value (auto/min/max): ")
                    if mode_input.lower() in {'auto', 'min', 'max'}:
                        return mode_input.lower()
                    else:
                        raise ValueError("Invalid mode selected. Please choose one of 'auto', 'min', or 'max'.\n")
                elif custom_mode.lower() == 'n':
                    print("Choosing the default mode: 'auto'\n")
                    return self.mode  # Use the default mode
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)
    
    """ Prompt the user for a custom verbose and validate the input."""
    def get_verbose(self):
        while True:
            try:
                custom_verbose = input("Do you want to choose a custom verbosity level? (Y:Yes /N:No) (Default: 1): ")
                if custom_verbose.lower() == 'y':
                    verbose_input = input("Enter the verbosity level (0= Silent, 1= Progress bar, 2= One line per epoch): ")
                    if verbose_input.strip() == '':
                        raise ValueError("Verbosity level cannot be empty. Please enter 0, 1, or 2.\n")
                    else:
                        verbose = int(verbose_input)
                        if verbose not in [0, 1, 2]:
                            raise ValueError("Invalid verbosity level. Please enter 0, 1, or 2.\n")
                        else:
                            return verbose
                elif custom_verbose.lower() == 'n':
                    print("Choosing the default verbosity level: (1)\n")
                    return self.verbose  # Use the default verbosity level
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)
    
    """Prompt the user for choosing process-based threading and validate the input."""
    def get_multiprocessing(self):
        while True:
            try:
                custom_multiprocessing = input("Do you want to use process-based threading? (Y:Yes /N:No) (Default: multiprocessing): ")
                if custom_multiprocessing.lower() == 'y':
                    return True
                elif custom_multiprocessing.lower() == 'n':
                    print("Choosing the default option: multiprocessing\n")
                    return False  # Use the default option
                else:
                    raise ValueError("Invalid input. Please enter (Y:Yes /N:No).\n")
            except ValueError as e:
                print(e)

    """Return the appropriate metrics based on the machine learning type."""
    def get_metrics(self):
        if self.ml_type == 'classifier':
            metrics = ['accuracy']
            return metrics
        elif self.ml_type == 'regressor':
            metrics = ['mse']
            return metrics
        else:
            raise ValueError("Invalid ml_type. Supported types are 'classifier' and 'regressor'.")
        
