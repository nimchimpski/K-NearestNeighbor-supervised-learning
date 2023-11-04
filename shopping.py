import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB

TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    # print(f"---len(evidence): {len(evidence)} ")
    # print(f"---labels: {labels} ")
    # print(f"---labels type: {type(labels)} ")
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )
    # print(f"---length X_train: {len(X_train)} length y_train: {len(y_train)} ")
    # print(f"---length X_test: {len(X_test)} length y_test: {len(y_test)} ")
    # print(f"---X_train: {X_train} y_train: {y_train}")
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
    """
    Load shopping data from a CSV file `filename` and convert into a list of
    evidence lists and a list of labels. Return a tuple (evidence, labels).

    evidence should be a list of lists, where each list contains the
    following values, in order:
        - Administrative, an integer
        - Administrative_Duration, a floating point number
        - Informational, an integer
        - Informational_Duration, a floating point number
        - ProductRelated, an integer
        - ProductRelated_Duration, a floating point number
        - BounceRates, a floating point number
        - ExitRates, a floating point number
        - PageValues, a floating point number
        - SpecialDay, a floating point number
        - Month, an index from 0 (January) to 11 (December)
        - OperatingSystems, an integer
        - Browser, an integer
        - Region, an integer
        - TrafficType, an integer
        - VisitorType, an integer 0 (not returning) or 1 (returning)
        - Weekend, an integer 0 (if false) or 1 (if true)

    labels should be the corresponding list of labels, where each label
    is 1 if Revenue is true, and 0 otherwise.
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader) # skip header

        evidence = []
        labels = []

        def month_to_int(month):
            if month == "Jan":
                return 0
            elif month == "Feb":
                return 1
            elif month == "Mar":
                return 2
            elif month == "Apr":
                return 3
            elif month == "May":
                return 4
            elif month == "June":
                return 5
            elif month == "Jul":
                return 6
            elif month == "Aug":
                return 7
            elif month == "Sep":
                return 8
            elif month == "Oct":
                return 9
            elif month == "Nov":
                return 10
            elif month == "Dec":
                return 11

        counter = 0
        for row in reader:
            counter += 1
            data = []
            data.append(int(row[0])),
            data.append(float(row[1])),
            data.append(int(row[2])),
            data.append(float(row[3])),
            data.append(int(row[4])),
            data.append(float(row[5])),
            data.append(float(row[6])),
            data.append(float(row[7])),
            data.append(float(row[8])),
            data.append(float(row[9])),
            data.append(int(month_to_int(row[10]))), # month_to_int
            data.append(int(row[11])),
            data.append(int(row[12])),
            data.append(int(row[13])),
            data.append(int(row[14])),
            data.append(1 if row[15] == "Returning_Visitor" else 0), # 0 or 1)
            data.append(1 if row[16] == "TRUE" else 0), # 0 or 1

            
            evidence.append(data)
            labels.append(1 if row[17] == "TRUE" else 0)

        # print(f"---counter: {counter}")
        return (evidence, labels)

def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    # print(f"+++train_model()")
    model =  KNeighborsClassifier(n_neighbors=1)
    # model = svm.SVC()
    # model = Perceptron()
    # model = GaussianNB()
    print(f"---model: {model}")
    return model.fit(evidence, labels)


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.

You may assume that the list of true labels will contain at least one positive label and at least one negative label.
    """
    # print(f"+++evaluate()")
    positives = 0
    sensitivity = 0
    negatives = 0
    specificity = 0
    for label, prediction in zip(labels, predictions):
        # print(f"---label: {label} prediction: {prediction}")
        if label == 1:
            positives += 1
            if prediction == 1:
                sensitivity += 1
        elif label == 0:
            negatives += 1
            if prediction == 0:
                specificity += 1
    sensitivity = sensitivity/positives
    specificity = specificity/negatives
    # print(f"---sensitivity: {sensitivity} specificity: {specificity}")
    return (sensitivity, specificity)
  


    # return sensitivity, specificity # floats


if __name__ == "__main__":
    main()
