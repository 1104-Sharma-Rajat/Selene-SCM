# Rajat Sharma
# File: main.py
# Date: 11/02/2025
# Description: Predicts a possible condition based on user-input symptoms using a trained ML model.

import pandas
from src.modelTrain import train_classifier
from colorama import Fore, Style

def main():
    while True:
        # Clear the terminal and print a simple colored header for the symptom classifier.
        print("\033c", end="")
        print(f"{Fore.CYAN}============================")
        print(f"{Fore.GREEN}     SYMPTOM CLASSIFIER")
        print(f"{Fore.CYAN}============================{Style.RESET_ALL}\n")

        # Load the symptom dataset and compute the feature list (exclude the diagnosis column).
        df = pandas.read_csv("data/trainingData.csv")
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
            if 1 <= i <= len(features ):
                data[features[i - 1]] = 1
        num_of_symptoms = len(chosen)

        # Convert our input dict into a DataFrame and get the model's prediction.
        user_df = pandas.DataFrame([data])
        probs = model.predict_proba(user_df)[0]
        classes = model.classes_

        ranked = sorted(
            zip(classes, probs),
            key=lambda x: x[1],
            reverse=True
        )

        # Print the predicted condition in a clean colored block for the user.
        print(f"\n{Fore.GREEN}============================")
        print(f"{Fore.GREEN}          Results")
        print(f"{Fore.GREEN}============================{Style.RESET_ALL}\n")

        top_label, top_prob = ranked[0]

        print(f"{Fore.CYAN}Most Likely Condition:{Style.RESET_ALL}")
        print(
            f"{Fore.WHITE}{top_label}{Style.RESET_ALL} "
            f"({Fore.GREEN}{top_prob * 100:.1f}%{Style.RESET_ALL})\n"
        )


        print(f"{Fore.YELLOW}Other Possible Conditions:{Style.RESET_ALL}")
        for label, prob in ranked[1:4]:
            print(f"{label} ({prob * 100:.1f}%)")

        if num_of_symptoms <= 2:
            print(f"\n{Fore.RED}Note: With only a few symptoms selected, the prediction may be less reliable.{Style.RESET_ALL}\n")

        print(
            f"\n{Fore.RED}This tool is for educational and informational purposes only.\n"
            f"It is NOT a substitute for professional medical advice, diagnosis, or treatment.\n"
            f"Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.\n\n"

            f"In the event of a medical emergency, call emergency services or seek immediate medical attention.{Style.RESET_ALL}"
        )


        restart = input(f"{Fore.MAGENTA}Check another case? (y/n): {Style.RESET_ALL}")
        if restart != "y" and restart != "Y":
            print(f"\n{Fore.CYAN}Bye for now!{Style.RESET_ALL}\n")
            break


if __name__ == "__main__":
        main()
