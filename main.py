# Rajat Sharma
# File: main.py
# Date: 11/02/2025
# Description: Predicts a possible condition based on user-input symptoms using a trained ML model.

import pandas
from src.modelTrain import train_classifier
from colorama import Fore, Style

def main():
    # Clear the terminal and print a simple colored header for the symptom classifier.
    print("\033c", end="")
    print(f"{Fore.CYAN}============================")
    print(f"{Fore.GREEN}     SYMPTOM CLASSIFIER")
    print(f"{Fore.CYAN}============================{Style.RESET_ALL}\n")

    # Load the symptom dataset and compute the feature list (exclude the diagnosis column).
    df = pandas.read_csv("data/symptomData.csv")
    features = [col for col in df.columns if col != "diagnosis"]

    # Print the available symptoms with indexes so the user knows what to choose.
    print(f"{Fore.YELLOW}Available Symptoms:{Style.RESET_ALL}")
    for i, name in enumerate(features, start=1):
        print(f"  {Fore.CYAN}{i:2d}{Style.RESET_ALL}. {name}")

    # Prompt the user to enter symptom numbers separated by commas.
    user_input = input(f"\n{Fore.MAGENTA}Enter symptom numbers (comma separated): {Style.RESET_ALL}")

    # Parse the user input into integers and bail out if formatting is incorrect.
    try:
        chosen = [int(x) for x in user_input.split(",")]
    except ValueError:
        print(f"{Fore.RED}Invalid input. Please enter numbers only.{Style.RESET_ALL}")
        return

    # Train (or load) the classifier and get the exact feature order expected by the model.
    print(f"\n{Fore.BLUE}Training model... please wait.{Style.RESET_ALL}")
    model, feature_names = train_classifier()

    # Build a single-row input dict initialized to zeros, then set chosen symptoms to 1.
    data = {f: 0 for f in feature_names}
    for i in chosen:
        if 1 <= i <= len(features):
            data[features[i - 1]] = 1

    # Convert our input dict into a DataFrame and get the model's prediction.
    user_df = pandas.DataFrame([data])
    prediction = model.predict(user_df)[0]

    # Print the predicted condition in a clean colored block for the user.
    print(f"\n{Fore.GREEN}------------------------------")
    print(f"{Fore.WHITE}Predicted Condition: {Fore.CYAN}{prediction}")
    print(f"{Fore.GREEN}------------------------------{Style.RESET_ALL}\n")

if __name__ == "__main__":
    main()
