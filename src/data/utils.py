from sklearn.metrics import (
    roc_curve,
    precision_score,
    roc_auc_score,
    recall_score,
    auc,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split


def getKmers(sequence, size=6):
    """Splitting the sequence into k-mers of a specified size
    Args:
        sequence: sequence to be split in k-mers
        size (default 6): size of k-mer
    Returns:
        list of list of k-mers (lower-cased)
    """
    return [sequence[x : x + size].lower() for x in range(len(sequence) - size + 1)]


def random_split(data_obj):
    """Randomly plitting the dataset object (see dataloader.py) into training and validation sets
    Args:
        data_obj: dataloader of which to split data
    Returns:
        randomly splitted datasets returned as training input, testing input, training labels, testing labels
    """
    X_train, X_test, y_train, y_test = train_test_split(
        data_obj.x, data_obj.y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def model_eval(predictions, ground_truth, predict_probas):
    """Assesing the model's performance on key metrics
    Args:
        predictions: binary classification predictions for each input sequence
        ground_truth: ground truth for each input sequence
        predict_probas: model prediction of probability to be classified as 1 for each input sequence
    Returns:
        roc_auc: computed Area Under the Receiver Operating Characteristic Curve (ROC AUC) from predictions
        auprc_score: computed Area Under the Curve (AUC) of Recall and Precision from predictions
    """
    fpr, tpr, threshold = roc_curve(
        ground_truth, predictions, pos_label=1
    )  # fpr: inc false positive rate; tpr: inc true positive rate
    prec = precision_score(ground_truth, predictions)
    rec = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    roc_auc = roc_auc_score(ground_truth, predict_probas)
    # auc_score = auc(fpr, tpr) # almost same as roc_auc, but possibly a bit too optimitic
    precision, recall, thresholds = precision_recall_curve(ground_truth, predict_probas)
    auprc_score = auc(recall, precision)
    print(
        "########### Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, AUROC: {3:.4f}, AUPRC: {4:.4f} ###########".format(
            prec, rec, f1, roc_auc, auprc_score
        )
    )

    return roc_auc, auprc_score
