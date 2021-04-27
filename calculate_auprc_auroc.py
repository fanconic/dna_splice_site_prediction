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
import pickle


def calc_auprc_auroc(model, ground_truth, predict_probas):
    """Prints the AUPRC and AUROC values
    Args:
        model: model name
        ground_truth: array of true labels
        predict_probas: test set prediction probablities
    """

    # here are the x, y scores for the AUPRC
    precision, recall, _ = precision_recall_curve(ground_truth, predict_probas)

    # here are the x, y scores for the AUROC
    fpr, tpr, _ = roc_curve(ground_truth, predict_probas)

    auprc_score = auc(recall, precision)
    auroc_score = auc(fpr, tpr)

    auprc_score_alt = average_precision_score(ground_truth, predict_probas)

    print("Model: {}".format(model))
    print(
        "AUPRC: {0:.4f},\tAverage Precision: {1:.4f},\tAUROC: {2:.4f}\n".format(
            auprc_score, auprc_score_alt, auroc_score, 
        )
    )


df_results = pickle.load(open("results.pkl", "rb"))

df_celegans = pd.read_csv(data_path + "C_elegans_test_seq.csv")
df_humans = pd.read_csv(data_path + "human_dna_test_split.csv")

for i in range(len(df_results)):
    row = df_results.iloc[i]
    data_name = row["input_data"]
    model = row["model_name"]
    if data_name == "C. elegans":
        print("C. elegans:")
        calc_auprc_auroc(model, df_celegans["labels"], row["test_predictions"])
    else:
        print("Human:")
        calc_auprc_auroc(model, df_humans["labels"], row["test_predictions"])
