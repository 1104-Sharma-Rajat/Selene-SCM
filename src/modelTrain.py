# Rajat Sharma
# File: modelTrain.py
# Date: 11/2/2025
# Description: Trains the machine learning model using the symptom dataset.


# Import the pandas library for working with the data, and RandomForestClassifier from sklearn for building the model.
# Additionally, import train_test_split for splitting the dataset and accuracy_score, classification_report for evaluating the model.
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(fp):
    inFile = pandas.read_csv(fp)
    X_Var = inFile.drop("diagnosis", axis=1)
    Y_Var = inFile["diagnosis"]
    return X_Var, Y_Var
    # Read the dataset from the csv file, utilizing pandas and converting to dataframe.
    # Since "diagnosis" is the target variable, we drop it from the feature set.
    # The target variable is stored in Y_Var.

def train_classifier():
    X_training, Y_training = load_data("data/trainingData.csv")
    X_validation, Y_validation = load_data("data/validationData.csv")
    X_testing, Y_testing = load_data("data/testData.csv")
    # Data split into training, validation, and testing sets.

    RF_Model = RandomForestClassifier(n_estimators = 200, random_state = 42, max_depth = 8, class_weight='balanced')
    # Initialize the RandomForestClassifier with 200 trees, a random state of 42, and a maximum depth of 8.
    # A random state of 42 ensures reproducibility of results.

    RF_Model.fit(X_training, Y_training)
    # Fit the model on the training data. Fitting represents the process of learning from the data.

    validation_prediction = RF_Model.predict(X_validation)
    print ("Validation Set Accuracy:", accuracy_score(Y_validation, validation_prediction))


    test_prediction = RF_Model.predict(X_testing)
    print ("Test Set Accuracy:", accuracy_score(Y_testing, test_prediction))
    print("\nClassification Report:\n", classification_report(Y_testing, test_prediction))
    # Make predictions on the validation and testing sets, then print accuracy scores and a detailed classification.

    return RF_Model, list(X_training.columns)
    # Return the trained model and the list of feature names for future use.
