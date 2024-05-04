import os 
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import ConvergenceWarning
from data_processing import UserInput
from model_trainer import MyAnn
from model_support import *




fitted_model = None

"""
Display the menu options.
"""

def show_menu():
    
    print("")
    print("Menu:")
    print("1. Train a New Model")
    print("2. Save Model")
    print("3. Load Model")
    print("4. Model Loss")
    print("5. Plot Model Loss")
    print("6. Make Prediction")
    print("7. Plot Model Accuracy")
    print("8. Print Classification Report")
    print("9. Plot Predictions Scatter")
    print("10. Print (RMSE), (MAE) ")
    print("11. Plot Residual Error")
    print("12. Print class attributes")
    print("0. Exit")


def main():
    """
    Main function to run the program.
    """

    global fitted_model
    
    
    while True:
        show_menu()
        choice = input("Enter your choice (0-12): ")

        if choice == '0':
            print("Exiting program.")
            break
        
        elif choice == '1': # Train a new model
            print("Welcome to the ANN training app...")
            
            # Instantiate UserInput and run_all method
            user_input_instance = UserInput()
            user_input_instance.run_all()

            # instantiate MyANN class and run_all method
            my_ann = MyAnn(user_input_instance.data, user_input_instance.target_column,
                        user_input_instance.hidden_layer_sizes, user_input_instance.activation,
                        user_input_instance.loss, user_input_instance.optimizer, user_input_instance.batch_size,
                        user_input_instance.epochs, user_input_instance.monitor, user_input_instance.patience,
                        user_input_instance.mode, user_input_instance.verbose, user_input_instance.multiprocessing, user_input_instance.metrics )
            fitted_model = my_ann.train_model()[0]       
       
        elif choice == '2': # Save  Model
            if fitted_model is not None:
                while True:
                    file_name = input("Enter the filename to save the model (include .h5 extension): ")
                    if file_name.endswith('.h5'):
                        save_model(my_ann.fitted_model, file_name)
                        break
                    else:
                        print("Invalid filename format. Please include the .h5 extension.")
            else:
                print("No Trained model was found.")
                
        elif choice == '3': # Load  Model
            while True:
                file_name = input("Enter the filename of the saved model (include .h5 extension): ")
                if file_name.endswith('.h5'):
                    user_input_instance = UserInput()
                    user_input_instance.get_file_path()
                    user_input_instance.get_target_column()
                    fitted_model= load_model(file_name)
                    break
                else:
                    print("Invalid filename format. Please include the .h5 extension.")

        elif choice == '4': # Model Losses
            if  fitted_model is not None:
                loss_data = model_loss(fitted_model)
                print(loss_data)
            else:
                print("No Trained model was found.")

        elif choice == '5': # Plot Losses
            if fitted_model is not None:
                plot_model_loss(fitted_model)
            else:
                print("No Trained model was found.")
        
        elif choice == '6': # Make Prediction
            if fitted_model is not None:
                # Get feature names from the dataset
                feature_names = [col for col in user_input_instance.data.columns if col != user_input_instance.target_column]
                
                # Initialize an empty dictionary to store user inputs
                user_input_row = {}
                
                # Prompt the user to input values for each feature
                for feature in feature_names:
                    while True:
                        try:
                            value = input(f"Enter value for '{feature}': ")
                            # Validate the input to ensure it's in the correct format
                            user_input_row[feature] = float(value)  # Convert input to float
                            break  # Exit the loop if input is valid
                        except ValueError:
                            print("Invalid input. Please enter a valid numerical value.")

                # Validate the input row length
                if len(user_input_row) != len(feature_names):
                    print("Error: Input row does not contain all features.")
                else:
                    # Convert the dictionary to a DataFrame row
                    input_row_df = pd.DataFrame([user_input_row])
                    # Make prediction using the model
                    prediction = make_prediction(fitted_model, input_row_df)
                    print("Prediction:", prediction)
            else:
                print("No Trained model was found.")
        
        elif choice == '7': # Plot model accuracy
            if user_input_instance.ml_type == 'classifier':
                if fitted_model is not None:
                    plot_model_accuracy(fitted_model)
                else:
                    print("No Trained model was found.")
            else:
                print("This option is only applicable for classification cases. It does not work for regressions.")

        elif choice == '8': # Print classification report
            if user_input_instance.ml_type == 'classifier':
                if fitted_model is not None:
                    print_classification_report(fitted_model, my_ann.y_test, my_ann.X_test)
                else:
                    print("No Trained model was found.")
            else:
                print("This option is only applicable for classification cases. It does not work for regressions.")
        
        elif choice == '9': # Plot Predictions Scatter
            if user_input_instance.ml_type == 'regressor':
                if fitted_model is not None:
                    plot_predictions_scatter(fitted_model, my_ann.y_test, my_ann.X_test)
                else:
                    print("No Trained model was found.")
            else:
                print("This option is only applicable for regression cases.")

        elif choice == '10': # Print (RMSE), (MAE)
            if user_input_instance.ml_type == 'regressor':
                if fitted_model is not None:
                    print_errors(fitted_model, my_ann.y_test, my_ann.scaled_X_test)
                else:
                    print("No Trained model was found.")
            else:
                print("This option is only applicable for regression cases.")

        elif choice == '11': # Plot Residual Error
            if fitted_model is not None:
                if user_input_instance.ml_type == 'regressor':
                    plot_residual_error(fitted_model, my_ann.y_test, my_ann.scaled_X_test)
                else:
                    print("This option is only applicable for regression cases.")
            else:
                print("No Trained model was found.")

        elif choice == '12': # Print class Attributes
            if fitted_model is not None:
                print("")
                get_attributes(fitted_model, user_input_instance.data, user_input_instance.target_column)
            else:
                print("No Trained model was found.")


if __name__ == "__main__":
    main()
    
    