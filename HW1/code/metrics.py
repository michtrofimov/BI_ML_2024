import numpy as np


def compute_confusion_matrix(y_true, y_pred):
    """
    Computes the confusion matrix for classification
    Arguments:
    y_true, np array (num_samples) - true labels
    y_pred, np array (num_samples) - model predictions
    Returns:
    np array (2x2) - confusion matrix
    """

    # Convert the input arrays to binary format (1 for positive class, 0 for negative class)
    # Initialize the confusion matrix
    cm = np.zeros((2, 2))

    # Fill in the confusion matrix
    for i in range(len(y_true)):
        if y_true[i] == "1":
            if y_pred[i] == "1":
                cm[0, 0] += 1
            else:
                cm[1, 0] += 1
        else:
            if y_pred[i] == "1":
                cm[0, 1] += 1
            else:
                cm[1, 1] += 1

    return cm


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """

    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score

    """
    YOUR CODE IS HERE
    """
    cm = compute_confusion_matrix(y_true, y_pred)
    cm = cm + 0.0001
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm)
    precision = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    recall = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    f1 = 2 * (precision * recall) / (precision + recall)

    return round(precision, 2), round(recall, 2), round(f1, 2), round(accuracy, 2)


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """

    """
    YOUR CODE IS HERE
    """
    accuracy = np.sum(y_pred == y_true) / len(y_true)

    return round(accuracy, 2)


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    """
    YOUR CODE IS HERE
    """
    y_true_mean = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true_mean) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    """
    YOUR CODE IS HERE
    """
    mse = np.mean((y_true - y_pred) ** 2)
    return mse


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    """
    YOUR CODE IS HERE
    """
    mae = np.mean(np.abs(y_true - y_pred))
    return mae
