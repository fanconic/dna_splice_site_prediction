import numpy as np
import pandas as pd
from settings import *
from sklearn.metrics import (
    average_precision_score,
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
import pickle


def calc_auprc_auroc(model, ground_truth, predict_probas):
    """Prints the AUPRC and AUROC values
    Args:
        model: model name
        ground_truth: array of true labels
        predict_probas: test set prediction probablities
    Returns:
        auprc_score: AUPRC score
        auroc_score: AUROC score
        precision: precision values for AUPRC score
        recall: recall values for AUPRC score
        fpr: fpr values for AUROC score
        tpr: tpr values for AUROC score
    """

    # here are the x, y scores for the AUPRC
    precision, recall, _ = precision_recall_curve(ground_truth, predict_probas)

    # here are the x, y scores for the AUROC
    fpr, tpr, _ = roc_curve(ground_truth, predict_probas)

    auprc_score = auc(recall, precision)
    auroc_score = auc(fpr, tpr)

    auprc_score_alt = average_precision_score(ground_truth, predict_probas)
    auroc_score_alt = roc_auc_score(ground_truth, predict_probas)

    print("Model: {}".format(model))
    print(
        "AUPRC: {0:.4f},\tAverage Precision: {1:.4f},\tAUROC: {2:.4f},\tAUROC (alternative): {3:.4f}\n".format(
            auprc_score, auprc_score_alt, auroc_score, auroc_score_alt
        )
    )

    return auprc_score, auroc_score, precision, recall, fpr, tpr


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

data_names = ["C. elegans", "Human"]

columns = [
    "model_name",
    "input_data",
    "AUROC",
    "AUPRC",
    "test_predictions",
    "hidden_test_predictions",
    "fpr",
    "tpr",
    "precision",
    "recall",
]
df_result = pd.DataFrame(columns=columns)

df_celegans = pd.read_csv(data_path + "C_elegans_test_seq.csv")
df_humans = pd.read_csv(data_path + "human_dna_test_split.csv")

for model in models:
    for data_name in data_names:
        if data_name == "C. elegans":
            preds = np.load(results_dir + model + "_celegans_probas.npy").flatten()
            preds_hidden = np.nan
            auprc_score, auroc_score, precision, recall, fpr, tpr = calc_auprc_auroc(
                model, df_celegans["labels"], preds
            )
        else:
            preds = np.load(results_dir + model + "_humans_probas_test.npy").flatten()
            preds_hidden = np.load(results_dir + model + "_humans_probas.npy").flatten()
            auprc_score, auroc_score, precision, recall, fpr, tpr = calc_auprc_auroc(
                model, df_humans["labels"], preds
            )

        new_row = {
            "model_name": model,
            "input_data": data_name,
            "AUROC": auroc_score,
            "AUPRC": auprc_score,
            "test_predictions": preds,
            "hidden_test_predictions": preds_hidden,
            "fpr": fpr,
            "tpr": tpr,
            "precision": precision,
            "recall": recall,
        }

        df_result = df_result.append(new_row, ignore_index=True)

pickle.dump(df_result, open("results.pkl", "wb"))
