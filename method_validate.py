import input_preprocess
import numpy as np
import pandas as pd
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier
import random

def split_data_by_percentage(data, test_percentage):
    """
    Splits the data into training and testing sets.
    """
    X_train = []
    X_test = []
    y_train = []
    y_test = []

    # shuffle data
    import random
    random.shuffle(data)

    # split data
    split_index = int(len(data) * test_percentage)
    X_train = [data[i][0] for i in range(split_index)]
    X_train = [[float(value) for value in row] for row in X_train]
    X_test = [data[i][0] for i in range(split_index, len(data))]
    X_test = [[float(value) for value in row] for row in X_test]
    y_train = [data[i][1] for i in range(split_index)]
    y_train = [row[0] for row in y_train]
    y_test = [data[i][1] for i in range(split_index, len(data))]
    y_test = [row[0] for row in y_test]
    return [X_train, X_test, y_train, y_test]

def split_data_by_chunks(data, k):
    """
    Splits the data into k chunks.
    """
    X_k_sets = []
    Y_k_sets = []

    # shuffle data
    import random
    random.shuffle(data)

    # split data
    split_index = int(len(data) / k)
    for i in range(k):
        X_k_sets.append([data[j][0] for j in range(split_index * i, split_index * (i + 1))])
        X_k_sets[i] = [[float(value) for value in row] for row in X_k_sets[i]]
        Y_k_sets.append([data[j][1] for j in range(split_index * i, split_index * (i + 1))])
        Y_k_sets[i] = [row[0] for row in Y_k_sets[i]]

    return [X_k_sets, Y_k_sets] # K chunks of registers for inputs and K for outputs

def make_train_and_test_sets(classes, test_percentage):
    """Select train and test data for each class"""
    for class_name in classes:
        classes[class_name] = split_data_by_percentage(classes[class_name], test_percentage)

    """Merge train and test data for each class"""
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for class_name in classes:
        X_train += classes[class_name][0]
        X_test += classes[class_name][1]
        y_train += classes[class_name][2]
        y_test += classes[class_name][3]
    return [X_train, X_test, y_train, y_test]

def make_k_folds_per_class(classes, k):
    """
    Makes k folds for each class.
    """
    for class_name in classes:
        classes[class_name] = split_data_by_chunks(classes[class_name], k) # [X_k_sets, Y_k_sets]

    return classes 


def split_data_by_class(data):
    """
    Splits the data by class.
    """
    classes = {}
    for i in range(len(data)):
        register = data[i]
        if (register[-1][0] not in classes):
            classes[register[-1][0]] = []
        classes[register[-1][0]].append(register)

    return classes # {class_name: [register1, register2, ...]}
    
def run_classifiers(X_train, X_test, y_train, y_test, k_nn, distance_metric):
     # Initialize and train KNN Classifier
    knn_classifier = KNNClassifier(k_nn, distance_metric)
    knn_classifier.fit(X_train, y_train)

    # Predict class labels for test data
    predictions_knn = knn_classifier.predict(X_test)

    # Initialize and train Minimum Distance Classifier
    min_dist_classifier = MinimumDistanceClassifier(distance_metric)
    min_dist_classifier.fit(X_train, y_train)

    # Predict class labels for test data
    predictions_min_dist = min_dist_classifier.predict(X_test)

    return [predictions_knn, predictions_min_dist]

def print_confusion_matrix(y_test, predictions):
    """
    Prints the confusion matrix in a readable format.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, predictions, labels=np.unique(y_test))
    cm_df = pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test))
    print("Confusion Matrix:\n", cm_df)

def get_confusion_matrix(y_test, predictions):
    """
    Calculates the confusion matrix.
    """
    class_names = np.unique(y_test)
    confusion_matrix = []
    # Initialize matrix
    for class_name in class_names:
        confusion_matrix.append([0] * len(class_names))
    # Fill matrix
    for i in range(len(y_test)):
        real_class = y_test[i]
        predicted_class = predictions[i]
        real_class_index = list(class_names).index(real_class)
        predicted_class_index = list(class_names).index(predicted_class)
        confusion_matrix[real_class_index][predicted_class_index] += 1
    return np.array(confusion_matrix)

def print_metrics(accuracy, precision, error_score, recall, f1, y_test, predictions):
    """
    Prints metrics and confusion matrix for a classifier.
    """
    # Print metrics
    print(f"Accuracy: {round(accuracy,3)}")
    print(f"Precision: {round(precision,3)}")
    print(f"Error score: {round(error_score,3)}")
    print(f"Recall: {round(recall,3)}")
    print(f"F1: {round(f1,3)}")

    # Print confusion matrix
    print_confusion_matrix(y_test, predictions)
    print("\n")
    
def get_accuracy(y_test, predictions):
    """
    Calculates accuracy for a classifier without sklearn.
    """
    correct = 0
    for i in range(len(y_test)):
        if y_test[i] == predictions[i]:
            correct += 1
    return correct / float(len(y_test))

def get_precision(y_test, predictions):
    """
    Calculates precision for a classifier
    """
    cm = get_confusion_matrix(y_test, predictions)
    precision = 0
    for i in range(len(cm)):
        precision += cm[i][i] / sum(cm[i])
    return precision / len(cm)

def get_recall(y_test, predictions):
    """
    Calculates recall for a classifier
    """
    cm = get_confusion_matrix(y_test, predictions)
    recall = 0
    for i in range(len(cm)):
        recall += cm[i][i] / sum(cm[:, i])
    return recall / len(cm)

def get_f1(y_test, predictions):
    """
    Calculates f1 for a classifier
    """
    precision = get_precision(y_test, predictions)
    recall = get_recall(y_test, predictions)
    return 2 * (precision * recall) / (precision + recall)


def calculate_metrics(y_test, predictions, printing = True):
    """
    Calculates metrics and confusion matrix for a classifier.
    """
    # Calculate metrics
    accuracy = get_accuracy(y_test, predictions)
    precision = get_precision(y_test, predictions)
    error_score = 1 - accuracy
    recall = get_recall(y_test, predictions)
    f1 = get_f1(y_test, predictions)
    confusion_matrix = get_confusion_matrix(y_test, predictions)

    # print metrics
    if printing:
        print_metrics(accuracy, precision, error_score, recall, f1, y_test, predictions)

    return [accuracy, precision, error_score, recall, f1, confusion_matrix]


def train_and_test(data, test_percentage, k_nn = 3, distance_metric="euclidiana"):
    """
    Trains and tests the classifier.
    """
    # split data
    classes = split_data_by_class(data)
    [X_train, X_test, y_train, y_test] = make_train_and_test_sets(classes, test_percentage)

    # Run a classifier
    [predictions_knn, predictions_min_dist] = run_classifiers(X_train, X_test, y_train, y_test, k_nn, distance_metric)

    # Calculate metrics
    print("Metrics for KNN Classifier:")
    calculate_metrics(y_test, predictions_knn)
    print("Metrics for Minimum Distance Classifier:")
    calculate_metrics(y_test, predictions_min_dist)
   

def k_fold_cross_validation(data, k=5, k_nn = 3, distance_metric="euclidiana"):
    """
    Performs K fold cross validation.
    """
    # split data
    classes = split_data_by_class(data) # {class_name: [register1, register2, ...]}

    # make k folds per class
    classes = make_k_folds_per_class(classes, k) # {class_name: [X_k_sets, Y_k_sets]}

    # initialize metrics
    accuracy_knn = 0
    precision_knn = 0
    error_score_knn = 0
    recall_knn = 0
    f1_knn = 0
    accuracy_min_dist = 0
    precision_min_dist = 0
    error_score_min_dist = 0
    recall_min_dist = 0
    f1_min_dist = 0

    # Initialize average metrics 
    avg_accuracy_knn = 0
    avg_precision_knn = 0
    avg_error_score_knn = 0
    avg_recall_knn = 0
    avg_f1_knn = 0
    avg_accuracy_min_dist = 0
    avg_precision_min_dist = 0
    avg_error_score_min_dist = 0
    avg_recall_min_dist = 0
    avg_f1_min_dist = 0

    # perform k fold cross validation
    for test_chunk in range(k):
        X_train = []
        X_test = []
        y_train = []
        y_test = []
        for class_name in classes.keys(): # For each class
            X_test += classes[class_name][0][test_chunk] # I Take one chunk as test
            y_test += classes[class_name][1][test_chunk]
            for train_chunk in range(k):
                if train_chunk != test_chunk:
                    X_train += classes[class_name][0][train_chunk]
                    y_train += classes[class_name][1][train_chunk]

        # Run classifiers
        [predictions_knn, predictions_min_dist] = run_classifiers(X_train, X_test, y_train, y_test, k_nn, distance_metric)
        print("Metrics for Fold " + str(test_chunk + 1) + ":")
        # Calculate metrics knn
        [accuracy_knn, precision_knn, error_score_knn, recall_knn, f1_knn, confusion_matrix_knn] = calculate_metrics(y_test, predictions_knn)
        # Calculate metrics min dist
        [accuracy_min_dist, precision_min_dist, error_score_min_dist, recall_min_dist, f1_min_dist, confusion_matrix_min_dist] = calculate_metrics(y_test, predictions_min_dist)
        print("-------------------------------------------")
        # Add metrics to average
        avg_accuracy_knn += accuracy_knn
        avg_precision_knn += precision_knn
        avg_error_score_knn += error_score_knn
        avg_recall_knn += recall_knn
        avg_f1_knn += f1_knn
        avg_accuracy_min_dist += accuracy_min_dist
        avg_precision_min_dist += precision_min_dist
        avg_error_score_min_dist += error_score_min_dist
        avg_recall_min_dist += recall_min_dist
        avg_f1_min_dist += f1_min_dist
    # Calculate average metrics
    avg_accuracy_knn /= k
    avg_precision_knn /= k
    avg_error_score_knn /= k
    avg_recall_knn /= k
    avg_f1_knn /= k
    avg_accuracy_min_dist /= k
    avg_precision_min_dist /= k
    avg_error_score_min_dist /= k
    avg_recall_min_dist /= k
    avg_f1_min_dist /= k
    # Print average metrics
    print("Average metrics for KNN Classifier:")
    print_metrics(avg_accuracy_knn, avg_precision_knn, avg_error_score_knn, avg_recall_knn, avg_f1_knn, y_test, predictions_knn)
    print("Average metrics for Minimum Distance Classifier:")
    print_metrics(avg_accuracy_min_dist, avg_precision_min_dist,avg_error_score_min_dist, avg_recall_min_dist, avg_f1_min_dist, y_test, predictions_min_dist)

def     bootstrap_validation(data, num_iterations=100, k_nn=3, distance_metric="euclidiana"):
    """
    Performs bootstrap validation.
    """
    n = len(data)
    metrics_knn = []
    metrics_min_dist = []

    for _ in range(num_iterations):
        # Create bootstrap sample
        sample_indices = [random.randint(0, n-1) for _ in range(n)]
        bootstrap_sample = [data[i] for i in sample_indices]
        oob_indices = list(set(range(n)) - set(sample_indices))
        oob_sample = [data[i] for i in oob_indices]

        # Split bootstrap sample and out-of-bag sample
        X_train, y_train = zip(*[(row[0], row[1][0]) for row in bootstrap_sample])
        X_test, y_test = zip(*[(row[0], row[1][0]) for row in oob_sample])

        # Convert X_train and X_test to numeric arrays
        X_train = np.array(X_train).astype(float)
        X_test = np.array(X_test).astype(float)

        # Run classifiers and evaluate metrics
        [predictions_knn, predictions_min_dist] = run_classifiers(X_train, X_test, y_train, y_test, k_nn, distance_metric)
        metrics_knn.append(calculate_metrics(y_test, predictions_knn, printing=False))
        metrics_min_dist.append(calculate_metrics(y_test, predictions_min_dist, printing=False))

    # print the average metrics using print_metrics
    # Separate numeric metrics and confusion matrices
    numeric_metrics_knn = [m[:5] for m in metrics_knn]  # Assuming the first 5 metrics are numeric
    confusion_matrices_knn = [m[5] for m in metrics_knn]  # Assuming the 6th metric is a confusion matrix
    numeric_metrics_min_dist = [m[:5] for m in metrics_min_dist]  # Assuming the first 5 metrics are numeric
    confusion_matrices_min_dist = [m[5] for m in metrics_min_dist]  # Assuming the 6th metric is a confusion matrix

    # Calculate average metrics
    avg_accuracy_knn = np.mean([m[0] for m in numeric_metrics_knn])
    avg_precision_knn = np.mean([m[1] for m in numeric_metrics_knn])
    avg_error_score_knn = np.mean([m[2] for m in numeric_metrics_knn])
    avg_recall_knn = np.mean([m[3] for m in numeric_metrics_knn])
    avg_f1_knn = np.mean([m[4] for m in numeric_metrics_knn])
    avg_accuracy_min_dist = np.mean([m[0] for m in numeric_metrics_min_dist])
    avg_precision_min_dist = np.mean([m[1] for m in numeric_metrics_min_dist])
    avg_error_score_min_dist = np.mean([m[2] for m in numeric_metrics_min_dist])
    avg_recall_min_dist = np.mean([m[3] for m in numeric_metrics_min_dist])
    avg_f1_min_dist = np.mean([m[4] for m in numeric_metrics_min_dist])

    # Print average metrics
    print('________________________________')
    print("Average metrics for KNN Classifier:")
    print_metrics(avg_accuracy_knn, avg_precision_knn, avg_error_score_knn, avg_recall_knn, avg_f1_knn, y_test, predictions_knn)
    print("Average metrics for Minimum Distance Classifier:")
    print_metrics(avg_accuracy_min_dist, avg_precision_min_dist,avg_error_score_min_dist, avg_recall_min_dist, avg_f1_min_dist, y_test, predictions_min_dist)


# File and delimiter
FILE_NAME = 'DB_new.csv'
DELIMITATOR = ','

# Read and preprocess the data
document = input_preprocess.read_from_file(FILE_NAME, DELIMITATOR)
for content in document:
    content[-1] = content[-1].strip()
    content[0] = content[0].replace("ï»¿", "")

attributes_number = len(document[0])
patterns_number = len(document)
attributes = input_preprocess.label_attributes(document, attributes_number, patterns_number)
attributes_input, attributes_output = input_preprocess.default_selection(attributes)

# Print attribute data (optional, for verification)
input_preprocess.print_attr_data(attributes, attributes_input, attributes_output, attributes_number, patterns_number)

# Select a subset of data for training and testing
new_matrix = input_preprocess.select_subset(document, attributes_number, patterns_number, DELIMITATOR)

# Ask user for type of distance metric
distance_metric = input("Ingrese el tipo de distancia (euclidiana o manhattan): ")

# Ask for the K for Knn
k_nn = int(input("Ingrese el valor de K para Knn: "))

# Ask user for percentage of data to use for testing
test_percentage = float(input("Ingrese el porcentaje de datos a utilizar para entrenamiento (0.8): "))
train_and_test(new_matrix, test_percentage, k_nn, distance_metric)

# Ask user for k value for K fold cross validation
k = int(input("Ingrese el valor de K para K fold cross validation: "))
k_fold_cross_validation(new_matrix, k, k_nn, distance_metric)

# Ask user for number of iterations for bootstrap validation
num_iterations = int(input("Ingrese el numero de iteraciones para bootstrap validation: "))
bootstrap_validation(new_matrix, num_iterations, k_nn, distance_metric)



