import warnings
warnings.filterwarnings("ignore")



"""
MyAnn - Customized Artificial Neural Network for Training and Prediction

This class implements a customized Artificial Neural Network (ANN) for training and prediction tasks. 
The ANN architecture and training settings can be customized through various parameters provided during 
initialization.

Parameters:
    data (DataFrame): The input dataset for training the ANN.
    target_column (str): The name of the target column in the dataset.
    hidden_layer_sizes (tuple): Tuple containing the sizes of hidden layers in the ANN.
    activation (str): The activation function to be used in the hidden layers.
    loss (str): The loss function to be optimized during training.
    optimizer (str): The optimizer algorithm to be used during training.
    batch_size (int): The batch size for training the ANN.
    epochs (int): The number of epochs (iterations over the entire dataset) for training.
    monitor (str): The metric to monitor during training for early stopping.
    patience (int): The number of epochs with no improvement after which training will be stopped.
    mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity 
                monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored 
                has stopped increasing; in 'auto' mode, the direction is automatically inferred.
    verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    multiprocessing (bool): Whether to use process-based threading for parallelism.
    metrics (list): List of metrics to evaluate the model during training.

Methods:
    - train_model(): Trains the ANN model with the provided settings and data.
    
Attributes:
    - X_train: Feature variables of the training dataset.
    - y_train: Target variable of the training dataset.
    - X_test: Feature variables of the testing dataset.
    - y_test: Target variable of the testing dataset.
    - scaled_X_train: Scaled feature variables of the training dataset.
    - scaled_X_test: Scaled feature variables of the testing dataset.
    - fitted_model: Trained ANN model after training.

Usage Example:
    # Create an instance of MyAnn class
    ann = MyAnn(data, 'target_column', (64, 32), 'relu', 'mse', 'adam', 32, 100, 'val_loss', 5, 'auto', 1, True, ['accuracy'])
    
    # Train the model
    fitted_model, X_train, y_train, X_test, y_test, scaled_X_train, scaled_X_test = ann.train_model()
"""


class MyAnn:

    """
        Initialize the MyAnn class.

        Args:
            data (DataFrame): The input dataset for training the ANN.
            target_column (str): The name of the target column in the dataset.
            hidden_layer_sizes (tuple): Tuple containing the sizes of hidden layers in the ANN.
            activation (str): The activation function to be used in the hidden layers.
            loss (str): The loss function to be optimized during training.
            optimizer (str): The optimizer algorithm to be used during training.
            batch_size (int): The batch size for training the ANN.
            epochs (int): The number of epochs (iterations over the entire dataset) for training.
            monitor (str): The metric to monitor during training for early stopping.
            patience (int): The number of epochs with no improvement after which training will be stopped.
            mode (str): One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity 
                        monitored has stopped decreasing; in 'max' mode it will stop when the quantity monitored 
                        has stopped increasing; in 'auto' mode, the direction is automatically inferred.
            verbose (int): Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            multiprocessing (bool): Whether to use process-based threading for parallelism.
            metrics (list): List of metrics to evaluate the model during training.
        """

    def __init__(self, data, target_column, hidden_layer_sizes,
                       activation, loss, optimizer, batch_size, epochs,
                       monitor, patience, mode, verbose, multiprocessing, metrics ):
        
        self.data = data
        self.target_column = target_column
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epochs = epochs
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.multiprocessing = multiprocessing
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.scaled_X_train = None
        self.scaled_X_test = None
        self.fitted_model = None 
        self.metrics = metrics

    def train_model(self):
        """
        Train the ANN model with the provided settings and data.

        Returns:
            tensorflow.keras.models.Sequential: The trained ANN model.
            numpy.ndarray: Feature variables of the training dataset.
            numpy.ndarray: Target variable of the training dataset.
            numpy.ndarray: Feature variables of the testing dataset.
            numpy.ndarray: Target variable of the testing dataset.
            numpy.ndarray: Scaled feature variables of the training dataset.
            numpy.ndarray: Scaled feature variables of the testing dataset.
        """

        print("Training model with the following settings:")
        print(f"Target Column: {self.target_column}")
        print(f"Hidden Layer Sizes: {self.hidden_layer_sizes}")
        print(f"Activation method : {self.activation}")
        print(f"Loss function : {self.loss}")
        print(f"Optimizer : {self.optimizer}")
        print(f"Batch size : {self.batch_size}")
        print(f"Epochs number : {self.epochs}")
        print(f"Monitor metric : {self.monitor}")
        print(f"Patience : {self.patience}")
        print(f"Mode: {self.mode}")
        print(f"Verbose: {self.verbose}")
        print(f"Use multiprocessing: {self.multiprocessing}")
        print("")


        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import MinMaxScaler

        # Splitting data into training and testing sets
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        
        # Scaling features
        scaler = MinMaxScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense,Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        model = Sequential()
        
        # Input layer
        model.add(Dense(units=self.hidden_layer_sizes[0]*2, input_dim=X_train.shape[1], activation= self.activation))

        # Hidden layers
        for units in self.hidden_layer_sizes[0:]:
            if units > 0:  # Check if units is positive, indicating a dense layer
                model.add(Dense(units=units, activation= self.activation))
            else:  # Negative value indicates a dropout layer
                dropout_rate = abs(units)   # Convert negative value to dropout rate (percentage)
                model.add(Dropout(rate=dropout_rate))

        # output layer 
        model.add(Dense(units=1, activation= self.activation))

        # Compile the model
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics= self.metrics)

        early_stop = EarlyStopping(monitor=self.monitor,patience=self.patience,mode=self.mode)
        # model training 
        model.fit(x= scaled_X_train,y= y_train, batch_size=self.batch_size, epochs=self.epochs,
                        validation_data=(scaled_X_test, y_test), verbose=self.verbose,
                        callbacks=[early_stop])

        self.fitted_model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scaled_X_train = scaled_X_train
        self.scaled_X_test = scaled_X_test
        return self.fitted_model, self.X_train,  self.y_train, self.X_test, self.y_test, self.scaled_X_train, self.scaled_X_test
       

