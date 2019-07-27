# 1차 모델 테스트

## 모델 선정

학습에 사용된 데이터는 1차 전처리 과정에서 도출된 데이터를 사용하였다.

{% page-ref page="../undefined-2/undefined-3/1.md" %}

초기 모델 테스트인 만큼, 지도 학습 내에서 분류\(Classification\)에 해당되는 대부분의 모델을 선정하였다.  
랜덤 포레스트를 제외한 모든 하이퍼파라미터는 Default 값으로 선정하여 실행하였다.

사용한 모델은 다음과 같다.

```python
LogisticRegression
SVC
KNeighborsClassifier
GaussianNB
Perceptron
LinearSVC
SGDClassifier
DecisionTreeClassifier
RandomForestClassifier
```

## 실행 결과

```python
1번 째 모델 학습 시작-------------------
LogisticRegression
              precision    recall  f1-score   support

           0       0.88      1.00      0.94    173151
           1       0.00      0.00      0.00     23793

    accuracy                           0.88    196944
   macro avg       0.44      0.50      0.47    196944
weighted avg       0.77      0.88      0.82    196944

2번 째 모델 학습 시작-------------------
KNeighborsClassifier
              precision    recall  f1-score   support

           0       0.89      0.95      0.92    173151
           1       0.32      0.16      0.22     23793

    accuracy                           0.86    196944
   macro avg       0.61      0.56      0.57    196944
weighted avg       0.82      0.86      0.84    196944

3번 째 모델 학습 시작-------------------
GaussianNB
              precision    recall  f1-score   support

           0       0.88      0.98      0.93    173151
           1       0.15      0.03      0.05     23793

    accuracy                           0.86    196944
   macro avg       0.52      0.50      0.49    196944
weighted avg       0.79      0.86      0.82    196944

4번 째 모델 학습 시작-------------------
Perceptron
              precision    recall  f1-score   support

           0       0.88      1.00      0.94    173151
           1       0.00      0.00      0.00     23793

    accuracy                           0.88    196944
   macro avg       0.44      0.50      0.47    196944
weighted avg       0.77      0.88      0.82    196944

5번 째 모델 학습 시작-------------------
LinearSVC
              precision    recall  f1-score   support

           0       0.88      1.00      0.94    173151
           1       0.00      0.00      0.00     23793

    accuracy                           0.88    196944
   macro avg       0.44      0.50      0.47    196944
weighted avg       0.77      0.88      0.82    196944

6번 째 모델 학습 시작-------------------
SGDClassifier
              precision    recall  f1-score   support

           0       0.88      1.00      0.94    173151
           1       0.00      0.00      0.00     23793

    accuracy                           0.88    196944
   macro avg       0.44      0.50      0.47    196944
weighted avg       0.77      0.88      0.82    196944

7번 째 모델 학습 시작-------------------
RandomForestClassifier
              precision    recall  f1-score   support

           0       0.91      0.89      0.90    173151
           1       0.31      0.35      0.33     23793

    accuracy                           0.83    196944
   macro avg       0.61      0.62      0.62    196944
weighted avg       0.84      0.83      0.83    196944

7번 째 모델 학습 시작------------------
RandomForestClassifier
              precision    recall  f1-score   support

           0       0.91      0.89      0.90    173151
           1       0.31      0.35      0.33     23793

    accuracy                           0.83    196944
   macro avg       0.61      0.62      0.62    196944
weighted avg       0.84      0.83      0.83    196944
```

## 결과 해석 및 회고

결과적으로는 절망적이다... 당연히 지연 데이터 자체가 적기 때문에 대부분의 경우에서 지연이 아니라 높은 정확도를 보여주었지만, 중요한 precision 및 recall에서 만족스러운 결과를 얻지 못했다.

단, 랜덤 포레스트에서 가장 우수한 성능을 보여주었기 떄문에 해당 모델을 집중적으로 연구할 필요가 있어 보인다.

## 실행 코드

```python
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
```

