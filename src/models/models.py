from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm


# models: k_NN, Random Forest, SVM, Boosting, TODO regressor, ...

class k_NN:
	def __init__(self, n_neighbors=3):
		clf = KNeighborsClassifier(n_neighbors=n_neighbors)

class random_forest:
	def __init__(self, ):
		clf =  RandomForestClassifier(n_estimators= 200, max_depth = max_depth, random_state=seed)

class SVM:
	def __init__(self):
		clf = StandardScaler()

class Boosting:
	def __init__(self):
		clf = lightgbm.LGBMClassifier()
