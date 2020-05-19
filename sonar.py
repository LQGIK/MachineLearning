import csv
import numpy as np
from sklearn.model_selection import train_test_split

# Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import Perceptron


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificty).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    sensitivity = 0
    specificity = 0
    totalSensitivity = sum(labels)
    total = len(labels)
    totalSpecificity = total - totalSensitivity

    for i in range(total):

        # Count specificity
        if labels[i] == 0:
            if predictions[i] == 0:
                specificity += 1

        else:
            if predictions[i] == 1:
                sensitivity += 1

    specificity /= totalSpecificity
    sensitivity /= totalSensitivity
    return (sensitivity, specificity)

def load(filename):
    # Read in csv file
    with open("sonar.csv") as f:
        reader = csv.reader(f)
        next(reader)

        inputs = []
        labels = []
        for row in reader:

            # Convert to floats
            rowToApp = [float(i) for i in row[:60]]
            inputs.append(rowToApp)
            # Mine = 1
            if row[60] == "M":
                labels.append(1)
            # Rock = 0
            else:
                labels.append(0)

        return inputs, labels

def main():

    # Get inputs / labels
    inputs, labels = load("sonar.csv")


    # Split data and reshape
    x_train, x_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.4
    )


    # Test various models
    modelNames = ["Perceptron", "KNeighborsClassifier", "SVC"]
    models = [Perceptron(), KNeighborsClassifier(1), SVC()]

    # Iterate models
    for i in range(len(models)):

        model = models[i]
        modelName = modelNames[i]

        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        # Evaluate the model
        sensitivity, specificity = evaluate(y_test, predictions)

        # Print results
        print("Model: ", modelName)
        print(f"Correct: {round((y_test == predictions).sum() / len(predictions) * 100, 2)}%")
        print(f"Incorrect: {round((y_test != predictions).sum() / len(predictions) * 100, 2)}%")
        print(f"True Positive Rate: {100 * sensitivity:.2f}%")
        print(f"True Negative Rate: {100 * specificity:.2f}%")
        print("\n")



if __name__ == "__main__":
    main()