import input_preprocess
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier

def split_data(data, test_percentage):
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

# File and delimiter
FILE_NAME = 'maternal.csv'
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

# train (80) and test data (20)
[X_train, X_test, y_train, y_test] = split_data(new_matrix, 0.2)

# Initialize and train KNN Classifier
knn_classifier = KNNClassifier(k=3, distance_metric=distance_metric)
knn_classifier.fit(X_train, y_train)

# Predict class labels for test data
predictions_knn = knn_classifier.predict(X_test)

# print metrics
print("______________________________________________________________")
print("Metrics for KNN Classifier:")
knn_classifier.print_confusion_matrix(y_test, predictions_knn)
knn_classifier.print_evaluation_metrics(y_test, predictions_knn)


# Initialize and train Minimum Distance Classifier
min_dist_classifier = MinimumDistanceClassifier(distance_metric=distance_metric)
min_dist_classifier.fit(X_train, y_train)

# Predict class labels for test data
predictions_min_dist = min_dist_classifier.predict(X_test)

# print metrics
print("______________________________________________________________")
print("Metrics for Minimum Distance Classifier:")
min_dist_classifier.print_confusion_matrix(y_test, predictions_min_dist)    
min_dist_classifier.print_evaluation_metrics(y_test, predictions_min_dist)

