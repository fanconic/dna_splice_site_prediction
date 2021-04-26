import os
import getpass
from src.models.metrics import AUPRC

data = "celegans"  # either 'humans' or 'celegans'

username = getpass.getuser()

# Google Colab
if "COLAB_GPU" in os.environ:
    data_path = "/content/drive/MyDrive/datasets/ML4H_p2/exercise_data/"
    out_dir = "/content/drive/My Drive/datasets/ML4H_p2/"
    results_dir = "/content/drive/My Drive/datasets/ML4H_p2/saved_output"
    colab = True

# Personal Computer -> you might have to adjust the path
else:
    data_path = "./exercise_data/"
    out_dir = "./saved_models/"
    results_dir = "./saved_output/"
    colab = False

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Dataset
celegans_seq = "C_elegans_acc_seq.csv"
hum_seq_hidden = "human_dna_test_hidden_split.csv"
hum_seq_test = "human_dna_test_split.csv"
hum_seq_train = "human_dna_train_split.csv"
hum_seq_val = "human_dna_validation_split.csv"

# Cross Validation
n_folds = 3

# Whether Predictions on testing set (see train.py) -> set to True only after everything has been finalized
predictionOnTestingSet = True

# Preprocessing
under_sampling_perc = 0.5
over_sampling_perc = 0.2
smote_strategy = None  # None for balanced
onehot = True

# SpliceAI
epochs = 15
batch_size = 256
