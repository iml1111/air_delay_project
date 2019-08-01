from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
# col: 출발/도착 지연/비지연 등등 구분
def proc_rf(df, col, est = 100):
	print("Machine Learning...")
	#### Train, Valid Set 준비하기
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
	df_x = df
	for df_index, val_index in split.split(df_x, df_x[col]) :
	    df1 = df_x.iloc[df_index]
	    val1 = df_x.iloc[val_index]
	#학습데이터셋 스플릿하기
	df_y = df1['DLY']
	df_x = df1.drop(['DLY'], axis=1)
	#밸리드데이터셋 스플릿하기
	val_y = val1['DLY']
	val_x = val1.drop(['DLY'], axis=1)
	###### 학습하기
	random_forest = RandomForestClassifier(n_estimators=est)
	random_forest.fit(df_x, df_y)
	##### 예측결과
	Y_pred = random_forest.predict(val_x)
	print(random_forest)
	print(classification_report(val_y, Y_pred))

def proc_svm(df, col, random_state=0, gamma=0.01, C=1.0):
	print("Machine Learning...")
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
	df_x = df
	for df_index, val_index in split.split(df_x, df_x[col]) :
	    df1 = df_x.iloc[df_index]
	    val1 = df_x.iloc[val_index]
	#학습데이터셋 스플릿하기
	df_y = df1['DLY']
	df_x = df1.drop(['DLY'], axis=1)
	#밸리드데이터셋 스플릿하기
	val_y = val1['DLY']
	val_x = val1.drop(['DLY'], axis=1)

	#### 학습하기
	svm = SVC(kernel="sigmoid", 
		random_state=random_state, gamma=gamma, C=C)
	svm.fit(df_x, df_y)
	#### 예측결과
	Y_pred = svm.predict(val_x)
	print(svm)
	print(classification_report(val_y, Y_pred))
