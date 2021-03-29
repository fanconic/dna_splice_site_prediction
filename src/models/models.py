from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# models: k_NN, Random Forest (Ensemble), SVM, Boosting, [MORE, at least 6]

class k_NN():
	''' K-nearest neighbors classifier

	'''
	def __init__(self, n_neighbors=3):
		self.clf = KNeighborsClassifier(n_neighbors=n_neighbors)

	def fit(train_x, train_y):
		self.clf.fit(train_x,train_y)

class random_forest():
	''' Random forest classifier (ensemble)

	'''
	def __init__(self, ):
		self.clf =  RandomForestClassifier(n_estimators= 200, max_depth = max_depth, random_state=seed)

class SVM():
	''' Support vector classification

	'''
	def __init__(self):
		self.clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))

class Boosting():
	''' Gradient boosting model

	'''
	def __init__(self):
		self.clf = lightgbm.LGBMClassifier()
