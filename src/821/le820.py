from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import roc_curve, auc,confusion_matrix, roc_auc_score

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

def l_proc(df, est = 100):
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
	#return
	######################### DNN ##############
	df_x['rf_score'] = random_forest.predict(df_x)
	val_x['rf_score'] = random_forest.predict(val_x)
	model = keras.Sequential([
    keras.layers.Dense(90, activation='tanh', input_shape=(len(df_x.columns),)),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(45, activation='tanh', kernel_regularizer=keras.regularizers.l2(0.01)),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(15, activation='tanh'),
    # 마지막 Layer
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
	])
	model.compile(optimizer='adam',
	              loss='binary_crossentropy',
	              metrics=['accuracy',
	                      tf.keras.metrics.Precision(name='precision'),
	                      tf.keras.metrics.Recall(name='recall'),
	                      auroc])
	print("training....")
	model.fit(df_x, df_y, epochs=3000, batch_size=30000, validation_data=(val_x, val_y), verbose=2)
	print("evaluating...")
	Y_pred = model.predict(val_x)
	#loss_and_metrics = model.evaluate(val_x, val_y, batch_size=30000)
	#print(classification_report(val_y, pred))
	Y_pred = [1 if Y_pred[i]>0.5 else 0 for i in range(len(Y_pred))]
	score = classification_report(val_y, Y_pred)
	print(score)
	''' 랜덤포레스트만
	RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                       max_depth=None, max_features='auto', max_leaf_nodes=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=3, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=1000,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
              precision    recall  f1-score   support

           0       0.89      0.99      0.94    168366
           1       0.76      0.14      0.24     23259

    accuracy                           0.89    191625
   macro avg       0.83      0.57      0.59    191625
weighted avg       0.88      0.89      0.86    191625

[[0.9449223  0.0550777 ]
 [0.75878289 0.24121711]
 [0.94332556 0.05667444]
 ...
 [0.8762286  0.1237714 ]
 [0.85600784 0.14399216]
 [0.74560882 0.25439118]]
0.8559545780036244
	'''