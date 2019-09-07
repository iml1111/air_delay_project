# 팀 IML 제출 코드 명세 문서

# 전처리 관련 코드
학습에 사용될 데이터에 대하여 전처리 및 정수 인코딩을 수행하기 위한 코드이다.
학습 데이터의 경우, p_proc -> label -> l_proc 함수를 거쳐서 학습이 수행된다. 
## p_proc
학습 데이터의 전처리 수행을 하는 함수이다.
_________
```python
df = pd.read_csv(file, engine='python', encoding="euc-kr")
```
항공 운항 데이터 파일 불러오기
```python
df = df.loc[df['CNL'] == 'N']
```
결항인 기록 제거
```python
df = df[df['REG'].notnull()]
```
항공 등록기호가 존재하지 않는 데이터 제거
```python
df = df[ df['IRR'] == "N" ]
```
부정기편 데이터 제거
```python
df = df[ df['ARP'] != df['ODP']  ]
```
출발 공항과 도착 공항이 같은 데이터 제거
```python
df = df.drop(["REG",'IRR',"DRR","CNL","CNR"], axis = 1)
```
학습에 사용되지 않는(테스트 데이터에 없는) 칼럼 제거

## label
학습 데이터 및 테스트 데이터에서 수행되는 공통적인 전처리 과정 수행 및 정수 인코딩 을 수행하는 함수이다. p_proc 과정을 거친 학습데이터와 테스트 데이터가 합쳐져서 함께 수행된다.
____
```python
	STT_Hour = []
	STT_Minute = []
	ATT_Hour = []
	ATT_Minute = []
	STT = df['STT']
	ATT = df['ATT']
	sub = STT.str.split(':', expand = True)
	STT_Hour = sub.iloc[0:, 0]
	STT_Minute = sub.iloc[0:, 1]
	sub2 = ATT.str.split(':', expand = True)
	ATT_Hour = sub2.iloc[0:, 0]
	ATT_Minute = sub2.iloc[0:, 1]
	df['STT_H'] = STT_Hour.astype(int)
	df['STT_M'] = STT_Minute.astype(int)
	df['ATT_H'] = ATT_Hour.astype(int)
	df['ATT_M'] = ATT_Minute.astype(int)
	df_STT_Time = df['STT_H'] * 60 + df['STT_M']
	df_ATT_Time = df['ATT_H'] * 60 + df['ATT_M']
	df_H3 = df_ATT_Time - df_STT_Time
	df['Time'] = df_H3
```
각 시각 데이터를 시간, 분 단위로 나눠서 
실제시각과 계획시간 간의 차이를 구하여 추가 칼럼 생성
```python
df = df.loc[(df['Time'] >= 0) | (df['DLY'] != 'Y') | (df['ATT_H'] < 0) | (df['ATT_H'] > 3) | (df['STT_H'] < 22)]
```
시간 차이가 음수(조기 출발)이거나 일부 새벽의 극소수 데이터 제거
```python
	df['DLY'].loc[df['DLY'] == 'Y'] = 1
	df['DLY'].loc[df['DLY'] == 'N'] = 0
```
지연 여부 정수 인코딩
```python
df['ARP_ODP'] = df['ARP'] + df['ODP']
	arp = (df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').sum())/(df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').count())
	arp = arp[arp['DLY'] > 0.7]
	arp = arp.index
	for i in arp:
	    df['ARP_ODP'].loc[df['ARP_ODP']==i] = 1
	df['ARP_ODP'].loc[df['ARP_ODP']!=1] = 0
	df = df.drop(['ARP', 'ODP'], axis = 1)
```
출발 공항 및 도착 공항 데이터를 합쳐서 하나의 고유 데이터로 매핑후, 
각 출발 도착 공항 데이터 제거
```python
df['SAME_DAY'] = (df['SDT_YY']*10000 + df['SDT_MM']*100 + df['SDT_DD']).astype(str) + df['FLO']
```
REG 데이터를 대신 할 고유의 항공기 식별 값 데이터 생성
```python
df = df.drop(["REG",'IRR',"DRR","CNL","CNR"], axis = 1)
```
학습에 사용되지 않는(테스트 데이터에 없는) 칼럼 제거
```python
df['SDT_DY'].loc[df['SDT_DY'] == '월'] = 0
df['SDT_DY'].loc[df['SDT_DY'] == '화'] = 1
df['SDT_DY'].loc[df['SDT_DY'] == '수'] = 2
df['SDT_DY'].loc[df['SDT_DY'] == '목'] = 3
df['SDT_DY'].loc[df['SDT_DY'] == '금'] = 4
df['SDT_DY'].loc[df['SDT_DY'] == '토'] = 5
df['SDT_DY'].loc[df['SDT_DY'] == '일'] = 6
```
요일 데이터 정수 인코딩
```python
label_encoder = preprocessing.LabelEncoder()
df_y = label_encoder.fit_transform(df['FLO']) 
df['FLO'] = df_y.reshape(len(df_y), 1)
df_y = label_encoder.fit_transform(df['SAME_DAY']) 
df['SAME_DAY'] = df_y.reshape(len(df_y), 1)
df_y = label_encoder.fit_transform(df['AOD']) 
df['AOD'] = df_y.reshape(len(df_y), 1)
df_y = label_encoder.fit_transform(df['FLT']) 
df['FLT'] = df_y.reshape(len(df_y), 1)
```
각 칼럼 데이터 정수 인코딩
```python
df = df.drop(['SDT_YY','ATT','Time',"STT","ATT","ATT_H","ATT_M","STT_M"], axis = 1)
```
학습에 사용되지 않는(테스트 데이터에 없는) 칼럼 제거

# 모델 관련 코드
전처리가 완료된 데이터를 바탕으로 모델이 학습을 수행하는 코드이다. 
## l_proc
```python
SEED = 42
lgb_params = { ... }
```
모델 하이퍼 파라미터 정의
```python
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state=42)
    df_x = df
    for df_index, val_index in split.split(df_x, df_x['SDT_MM']):
        df1 = df_x.iloc[df_index]
        val1 = df_x.iloc[val_index]
    df_y = df1['DLY']
    df_x = df1.drop(['DLY'], axis=1)
    val_y = val1['DLY']
    val_x = val1.drop(['DLY'], axis=1)
```
학습 데이터 셋 및 밸리데이션 셋 나누기
```python
    tr_data = lgb.Dataset(df_x, label=df_y)
    vl_data = lgb.Dataset(val_x, label = val_y) 
```
lightbgm에서 제공하는 전용 데이터 셋 객체 생성
```python
estimator = lgb.train(
        lgb_params,
        tr_data,
        valid_sets = [tr_data, vl_data],
        verbose_eval = 200,
    )    
Y_pred = estimator.predict(val_x)
print(roc_auc_score(val_y, Y_pred))
```
지정된 하이퍼 파라미터와 데이터 셋을 입력해서 학습 및 성능 평가








