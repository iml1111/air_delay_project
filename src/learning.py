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

def proc(df):
	print("Machine Learning...")
	#### Train, Valid Set 준비하기
	# 테스트에없는 칼럼 삭제
	df = df.drop(['DRR','ATT_H',"ATT_M","SDT_YY","STT_M"], axis = 1)
	#데이터셋 스플릿하기
	from sklearn.model_selection import StratifiedShuffleSplit
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
	df_x = df
	for df_index, val_index in split.split(df_x, df_x['AOD']) :
	    df1 = df_x.iloc[df_index]
	    val1 = df_x.iloc[val_index]
	#학습데이터셋 스플릿하기
	df_y = df1['DLY']
	df_x = df1.drop(['DLY'], axis=1)
	#밸리드데이터셋 스플릿하기
	val_y = val1['DLY']
	val_x = val1.drop(['DLY'], axis=1)

	###### 학습하기
	models = [
	linear_model.LogisticRegression(),
	#SVC(),
	KNeighborsClassifier(n_neighbors = 3),
	GaussianNB(),
	Perceptron(),
	LinearSVC(),
	SGDClassifier(),
	DecisionTreeClassifier()
	]
	for idx, model in enumerate(models):
		print(idx+1, "번 째 모델 학습 시작-------------------")
		model.fit(df_x, df_y)
		pred = model.predict(val_x)
		print(model)
		print(classification_report(val_y, pred))
		print()

	random_forest = RandomForestClassifier(n_estimators=10000)
	random_forest.fit(df_x, df_y)
	Y_pred = random_forest.predict(val_x)
	print(len(models), "번 째 모델 학습 시작-------------------")
	print(random_forest)
	print(classification_report(val_y, Y_pred))
	print()