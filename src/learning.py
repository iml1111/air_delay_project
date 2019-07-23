######학습하기
from sklearn import linear_model
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings("ignore")

models = [
linear_model.LogisticRegression(),
SVC(),
KNeighborsClassifier(n_neighbors = 3),
GaussianNB(),
Perceptron(),
LinearSVC(),
SGDClassifier(),
DecisionTreeClassifier()
]
for model in models:
	model.fit(df_x, df_y)
	pred = model.predict(val_x)
	print(model)
	print(classification_report(val_y, pred))
	print()

random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(df_x, df_y)
Y_pred = random_forest.predict(val_x)
print(random_forest)
print(classification_report(val_y, pred))
print()