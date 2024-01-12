import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.spatial.distance import cdist
from collections import Counter
import pandas as pd

def euclidean_distance(a, b):
    """
    Computes the Euclidean distance between two vectors.
    """
    return np.linalg.norm(np.array(a) - np.array(b))

def manhattan_distance(a, b):
    """
    Computes the Manhattan distance between two vectors.
    """
    return np.sum(np.abs(np.array(a) - np.array(b)))

class KNNClassifier:
    def __init__(self, k=3, distance_metric='euclidean'):
        """
        Initializes the KNN classifier with the number of neighbors and distance metric.
        """
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        Fits the classifier with the training data.
        """
        self.X_train = X
        self.y_train = y

    def print_evaluation_metrics(self, y_true, y_pred):
        """
        Prints evaluation metrics including accuracy, precision, recall, and F1-score.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score:", f1)
        print("\nClassification Report:\n", classification_report(y_true, y_pred))
    
    def get_confusion_matrix(self, y_true, y_pred):
        """
        Calculates the confusion matrix.
        """
        return confusion_matrix(y_true, y_pred, labels=np.unique(y_true))

    def print_confusion_matrix(self, y_true, y_pred):
        """
        Prints the confusion matrix in a readable format.
        """
        cm = self.get_confusion_matrix(y_true, y_pred)
        cm_df = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
        print("Confusion Matrix:\n", cm_df)
    
    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        """
        predictions = []
        for x in X:
            # Compute distances from the test point to all training points
            distances = [euclidean_distance(x, x_train) if self.distance_metric == 'euclidean' 
                         else manhattan_distance(x, x_train) for x_train in self.X_train]

            # Sort by distance and get the indices of the nearest neighbors
            k_indices = np.argsort(distances)[:self.k]

            # Extract the labels of the nearest neighbors
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Determine the most common class label among the nearest neighbors
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions
