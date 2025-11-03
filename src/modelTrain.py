# Rajat Sharma
# File: modelTrain.py
# Date: 11/2/2025
# Description: Trains the machine learning model using the symptom dataset.


# Import the pandas library for working with the data, and DecisionTreeClassifier from sklearn for building the model.
# Additionally, import train_test_split for splitting the dataset and accuracy_score, classification_report for evaluating the model.
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_classifier():
    df = pandas.read_csv("data/symptomData.csv")
    # Read the dataset from the csv file, utilizing pandas and converting to dataframe.

    X_Var = df.drop("diagnosis", axis=1)
    # Since "diagnosis" is the target variable, we drop it from the feature set.
    Y_Var = df["diagnosis"]
    # The target variable is stored in Y_Var.

    X_train, X_test, Y_train, Y_test = train_test_split(X_Var, Y_Var, test_size = 0.2, random_state = 42)
    # Split the dataset into training and testing sets, with 20% of the data reserved for testing, and the 42 random state for reproducibility.

    DT_Model = RandomForestClassifier(n_estimators = 200, random_state = 42, max_depth = 8)
    # Initialize the RandomForestClassifier with 200 trees, a random state of 42, and a maximum depth of 8.
    # A random state of 42 ensures reproducibility of results.

    DT_Model.fit(X_train, Y_train)
    # Fit the model on the training data. Fitting represents the process of learning from the data.

    Y_Pred = DT_Model.predict(X_test)
    # Make predictions on the test set. This generates predicted labels for the test data.

    accuracy = accuracy_score(Y_test, Y_Pred)
    # Calculate the accuracy of the model by comparing the predicted labels with the actual labels from the test set.

    print("Model Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(Y_test, Y_Pred))

    return DT_Model, list(X_Var.columns)
    # Return the trained model and the list of feature names for future use.
