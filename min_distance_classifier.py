import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
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

class MinimumDistanceClassifier:
    def __init__(self, distance_metric='euclidean'):
        """
        Initializes the Minimum Distance classifier with the specified distance metric.
        """
        self.distance_metric = distance_metric
        self.means = {}
        self.classes = []

    def fit(self, X, y):
        """
        Fits the classifier with the training data.
        """
        self.classes = np.unique(y)
        for cls in self.classes:
            self.means[cls] = np.mean([X[i] for i in range(len(X)) if y[i] == cls], axis=0)

    def predict(self, X):
        """
        Predicts the class labels for the provided data.
        """
        predictions = []
        for x in X:
            # Compute distances from the test point to the mean of each class
            distances = {cls: euclidean_distance(x, mean) if self.distance_metric == 'euclidean' 
                         else manhattan_distance(x, mean) for cls, mean in self.means.items()}

            # Choose the class with the minimum distance
            predictions.append(min(distances, key=distances.get))
        return predictions

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
