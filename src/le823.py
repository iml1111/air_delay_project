from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc,confusion_matrix, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def l_proc(df, est = 100):
	#### Train, Valid Set 준비하기
	split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
	df_x = df
	for df_index, val_index in split.split(df_x, df_x['SDT_MM']):
	    df1 = df_x.iloc[df_index]
	    val1 = df_x.iloc[val_index]
	#학습데이터셋 스플릿하기
	df_y = df1['DLY']
	df_x = df1.drop(['DLY'], axis=1)
	#밸리드데이터셋 스플릿하기
	val_y = val1['DLY']
	val_x = val1.drop(['DLY'], axis=1)
	print("split done...")

	"""## Random Forest"""
	print("Machine Learning...")
	###### 학습하기
	random_forest = RandomForestClassifier(n_estimators=est, min_samples_leaf=3)
	random_forest.fit(df_x, df_y)
	##### 예측결과
	Y_pred = random_forest.predict(val_x)
	print(Y_pred)
	print(val_y)
	print(random_forest)
	print(classification_report(val_y, Y_pred))

	y_score = random_forest.predict_proba(val_x)
	print(y_score)
	print(roc_auc_score(val_y, y_score[:,1]))
	print('end')

def l2_proc(df):
	#### Train, Valid Set 준비하기
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
	print("split done...")

	"""## Random Forest"""
	print("Machine Learning...")
	###### 학습하기
	ax_features = ['auto', 'sqrt']
	min_samples_split = [2, 5, 10]
	min_samples_leaf = [1, 2, 3, 4]
	bootstrap = [True, False]
	random_grid = {#'n_estimators': n_estimators,
               'max_features': max_features,
               #'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               #'bootstrap': bootstrap
                }
	model = RandomizedSearchCV(random_forest, param_distributions=random_grid, scoring='roc_auc', random_state=42)
	Y_pred = model.fit(df_x, df_y).predict(val_x)
	print(Y_pred)
	print(val_y)
	print(model)
	print(classification_report(val_y, Y_pred))
	y_score = model.predict_proba(val_x)
	print(y_score)
	print(roc_auc_score(val_y, y_score[:,1]))
	print('end')

#기본: [Finished in 2607.9s]
#TO DO: 0.8501690514322111
#AOD로 나누는거 딴걸로 나눠보기
#L2 돌려보기
#이전 지연 예측 정보 해당 정보에 반영하기