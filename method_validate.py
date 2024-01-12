import input_preprocess
from knn_classifier import KNNClassifier
from min_distance_classifier import MinimumDistanceClassifier

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

def print_metrics(accuracy, precision, recall, f1, confusion_matrix):
    """
    Prints metrics and confusion matrix for a classifier.
    """
    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Print confusion matrix
    print("Confusion Matrix:")
    print(confusion_matrix)

def calculate_metrics(y_test, predictions, print=True):
    """
    Calculates metrics and confusion matrix for a classifier.
    """
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')

    # Calculate confusion matrix
    from sklearn.metrics import confusion_matrix
    confusion_matrix = confusion_matrix(y_test, predictions)

    print_metrics(accuracy, precision, recall, f1, confusion_matrix)

    return [accuracy, precision, recall, f1, confusion_matrix]


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
    calculate_metrics(y_test, predictions_knn)
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
    recall_knn = 0
    f1_knn = 0
    accuracy_min_dist = 0
    precision_min_dist = 0
    recall_min_dist = 0
    f1_min_dist = 0

    # Initialize average metrics 
    avg_accuracy_knn = 0
    avg_precision_knn = 0
    avg_recall_knn = 0
    avg_f1_knn = 0
    avg_accuracy_min_dist = 0
    avg_precision_min_dist = 0
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
        # Calculate metrics knn
        [accuracy_knn, precision_knn, recall_knn, f1_knn, confusion_matrix_knn] = calculate_metrics(y_test, predictions_knn, False)
        # Calculate metrics min dist
        [accuracy_min_dist, precision_min_dist, recall_min_dist, f1_min_dist, confusion_matrix_min_dist] = calculate_metrics(y_test, predictions_min_dist, False)
        # Add metrics to average
        avg_accuracy_knn += accuracy_knn
        avg_precision_knn += precision_knn
        avg_recall_knn += recall_knn
        avg_f1_knn += f1_knn
        avg_accuracy_min_dist += accuracy_min_dist
        avg_precision_min_dist += precision_min_dist
        avg_recall_min_dist += recall_min_dist
        avg_f1_min_dist += f1_min_dist
    # Calculate average metrics
    avg_accuracy_knn /= k
    avg_precision_knn /= k
    avg_recall_knn /= k
    avg_f1_knn /= k
    avg_accuracy_min_dist /= k
    avg_precision_min_dist /= k
    avg_recall_min_dist /= k
    avg_f1_min_dist /= k
    # Print average metrics
    print_metrics(avg_accuracy_knn, avg_precision_knn, avg_recall_knn, avg_f1_knn, confusion_matrix_knn)
    print_metrics(avg_accuracy_min_dist, avg_precision_min_dist, avg_recall_min_dist, avg_f1_min_dist, confusion_matrix_min_dist)

        


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
test_percentage = float(input("Ingrese el porcentaje de datos a utilizar para pruebas (0.2): "))
train_and_test(new_matrix, test_percentage, k_nn, distance_metric)

# Ask user for k value for K fold cross validation
k = int(input("Ingrese el valor de K para K fold cross validation: "))
k_fold_cross_validation(new_matrix, k, k_nn, distance_metric)



