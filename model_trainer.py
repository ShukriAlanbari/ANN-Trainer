import warnings
warnings.filterwarnings("ignore")






class MyAnn:

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
       

