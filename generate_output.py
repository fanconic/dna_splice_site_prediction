import numpy as np
import pandas as pd
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

# celegans results
df = pd.read_csv(data_path + "C_elegans_test_seq.csv")
df = df.drop(["labels"], axis=1)
df = df.rename(columns={"Unnamed: 0": "original_index"})

for model in models:
    probas = np.load(results_dir + model + "_celegans_probas.npy")
    df[model + "_preds"] = probas.flatten()
df.to_csv(results_dir + "test_celegans_results.csv")

# humans hidden
df = pd.read_csv(data_path + "human_dna_test_hidden_split.csv")
for model in models:
    probas = np.load(results_dir + model + "_humans_probas.npy")
    df[model + "_preds"] = probas.flatten()
df.to_csv(results_dir + "test_hidden_humans_results.csv")

# humans test
df = pd.read_csv(data_path + "human_dna_test_split.csv")
for model in models:
    probas = np.load(results_dir + model + "_humans_probas_test.npy")
    df[model + "_preds"] = probas.flatten()
df.to_csv(results_dir + "test_humans_results.csv")
