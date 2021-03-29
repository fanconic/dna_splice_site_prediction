import os
import getpass

username = getpass.getuser()

'''
if "COLAB_GPU" in os.environ:
    data_path = "/content/ml4h_proj1_colon_cancer_ct/"
    out_dir = "/content/drive/MyDrive/ML4H/"
    colab = True
else:
    data_path = "/cluster/scratch/{}/ML4H/ML4H_proj_2/exercise_data/".format(username)
    out_dir = "/cluster/scratch/{}/ML4H/ML4H_proj_2/saved_models/".format(username)
    colab = False
'''
data_path = '/home/manu/ethz_master/FS21/ML4H/project_2/exercise_data/'

# Dataset

celegans_seq = "C_elegans_acc_seq.csv"
hum_seq_hidden = "human_dna_test_hidden_split.csv"
hum_seq_test = "human_dna_test_split.csv"
hum_seq_train = "human_dna_train_split.csv"
hum_seq_val = "human_dna_validation_split.csv"

# Preprocessing
under_sampling_perc = 0.5
over_sampling_perc = 0.2
smote_strategy = None #None for balanced


# Boosting



# SVM


# k-NN
n_neighbors = 3



# Random Forest
max_depth = 10
seed = 42
n_estimators = 200
