import sys
import pandas
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime


TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):

    # Load the data
    df = pandas.read_csv(filename)

    # The titles that need to be converted to integers
    to_int = [
        "Administrative",
        "Informational",
        "ProductRelated",
        "OperatingSystems",
        "Browser",
        "Region",
        "TrafficType",
        "Weekend",
        "Revenue",
    ]

    # The titles that need to be converted to floats
    to_float = [
        "Administrative_Duration",
        "Informational_Duration",
        "ProductRelated_Duration",
        "BounceRates",
        "ExitRates",
        "PageValues",
        "SpecialDay",
    ]

    # Convert
    df[to_int] = df[to_int].astype(int)
    df[to_float] = df[to_float].astype(float)

    # Convert the visitor type column to 0 or 1
    df["VisitorType"] = df["VisitorType"].apply(
        lambda x: 1 if x == "Returning_Visitor" else 0
    )

    # Making month column consistent
    df["Month"] = df["Month"].apply(lambda x: x.replace("June", "Jun"))

    # Convert the month column to an index from 0 to 11
    df["Month"] = df["Month"].apply(
        lambda old_month: datetime.strptime(old_month, "%b").month - 1
    )

    evidences = df.iloc[:, :-1]
    labels = df["Revenue"]

    return evidences, labels


def train_model(evidence, labels):
    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(evidence, labels)
    return clf


def evaluate(labels, predictions):

    # using confusion matrix to calculate the sensitivity and specificity
    matrix = confusion_matrix(labels, predictions)

    sensitivity = matrix[1][1] / (matrix[1][0] + matrix[1][1])
    specificity = matrix[0][0] / (matrix[0][0] + matrix[0][1])

    return sensitivity, specificity


if __name__ == "__main__":
    main()
