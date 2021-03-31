from sklearn.metrics import (
    roc_curve,
    precision_score,
    roc_auc_score,
    recall_score,
    auc,
    f1_score,
)
from sklearn.model_selection import train_test_split


def getKmers(sequence, size=6):
    """ Splitting the sequence into k-mers of a specified size

	"""
    return [sequence[x : x + size].lower() for x in range(len(sequence) - size + 1)]


def random_split(data_obj):
    """ Splitting the dataset object (see dataloader.py) into training and validation sets

	"""
    X_train, X_test, y_train, y_test = train_test_split(
        data_obj.x, data_obj.y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def model_eval(predictions, ground_truth):
    """ Assesing the model's performance on key metrics

	"""
    fpr, tpr, threshold = roc_curve(
        ground_truth, predictions, pos_label=1
    )  # fpr: inc false positive rate; tpr: inc true positive rate
    prec = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)
    f1 = f1_score(ground_truth, predictions)
    roc_auc = roc_auc_score(ground_truth, predictions)
    auc_score = auc(fpr, tpr)

    print(
        "########### Precision : {0:.4f}, Recall: {1:.4f}, F1: {2:.4f}, AUROC: {3:.4f}, AUPRC: {4:.4f} ###########".format(
            prec, recall, f1, roc_auc, auc_score
        )
    )

    return auc_score
