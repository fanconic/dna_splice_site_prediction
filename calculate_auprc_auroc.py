import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from settings import *

models = [
    "K-Nearest Neighbours",
    "Logistic Regression",
    "Linear Support Vector Machine",
    "Support Vector Machine",
    "Gradient Boosting",
    "MLP",
    "Random Forest",
    "SpliceAI80",
    "SpliceAI400",
]


# Human Test set
df_human_test_predictions = pd.read_csv(results_dir + "test_humans_results.csv")
df_human_test_labels = pd.read_csv(data_path + "human_dna_test_split.csv")

# C. elegans
df_celegans_test_predictions = pd.read_csv(results_dir + "test_celegans_results.csv")
df_celegans_test_labels = pd.read_csv(data_path + "C_elegans_test_seq.csv")

# @ TA: you can uncomment the following lines and enter your path to the hidden human dataset
# df_celegans_hidden_test_predictions = pd.read_csv(results_dir + "test_hidden_humans_results.csv")
# df_celegans_hidde_test_labels = pd.read_csv(...)


def print_results(models, ground_truth, df_preds):
    """Prints the AUPRC and AUROC values
    Args:
        models: list of model names
        ground_truth: array of true labels
        df_preds: submission data frame with prediction probablilities
    """
    for model in models:
        predict_probas = df_preds[model + "_preds"]

        # here are the x, y scores for the AUPRC
        precision, recall, _ = precision_recall_curve(ground_truth, predict_probas)

        # here are the x, y scores for the AUROC
        tpr, fpr, _ = roc_curve(ground_truth, predict_probas)

        auprc_score = auc(recall, precision)
        auroc_score = auc(tpr, fpr)

        auprc_score_alt = average_precision_score(ground_truth, predict_probas)
        auroc_score_alt = roc_auc_score(ground_truth, predict_probas)

        print("Model: {}".format(model))
        print(
            "AUPRC: {0:.4f},\tAverage Precision: {1:.4f},\tAUROC: {2:.4f},\tAUROC (alternative): {3:.4f}\n".format(
                auprc_score, auprc_score_alt, auroc_score, auroc_score_alt
            )
        )


print("Human Test Set")
ground_truth = df_human_test_labels["labels"]
print_results(models, ground_truth, df_human_test_predictions)

# @ TA: uncomment here to run the predicitons
"""
print("Human Hidden Test Set")
ground_truth = df_celegans_hidde_test_labels["labels"]
print_results(models, ground_truth, df_celegans_hidden_test_predictions)
"""

print("C. Elegans Test Set")
ground_truth = df_celegans_test_labels["labels"]
print_results(models, ground_truth, df_celegans_test_predictions)
