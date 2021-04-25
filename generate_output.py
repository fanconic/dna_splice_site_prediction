import numpy as np
import pandas as pd
from settings import *

# celegans results
df = pd.read_csv(data_path + "C_elegans_test_seq.csv")
preds = np.load(results_dir + "Gradient Boosting_celegans_results.npy")
probas = np.load(results_dir + "Gradient Boosting_celegans_probas.npy")
df = df.drop(["labels"], axis=1)
df = df.rename(columns={"Unnamed: 0": "original_index"})
df["predicitons"] = preds.astype(int)
df["predicitons"] = df["predicitons"].replace(0, -1)
df["prediction_probabilities"] = probas
df.to_csv(results_dir + "celegans_results.csv")

# humans
df = pd.read_csv(data_path + "human_dna_test_hidden_split.csv")
preds = np.load(results_dir + "SpliceAI400_humans_results.npy")
probas = np.load(results_dir + "SpliceAI400_humans_probas.npy")
df["predicitons"] = preds.flatten().astype(int)
df["predicitons"] = df["predicitons"].replace(0, -1)
df["prediction_probabilities"] = probas.flatten()
df.to_csv(results_dir + "humans_results.csv")
