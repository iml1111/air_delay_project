import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
from sklearn import preprocessing

def proc(filename = 'AFSNT.csv'):
	###############파일 불러오기
	print("file_load...")
	df = pd.read_csv('AFSNT.csv', engine='python')

	#############전처리하기
	print("pre-processing...")
	#널제거
	#df = df.fillna("null")
	#ATT 빈값 STT로 채우기
	guess_df = df['STT'].loc[df['ATT'].isnull()]
	df['ATT'].loc[df['ATT'].isnull()] = guess_df
	# 결항처리제거하고 결항 칼럼 날리기
	df = df.loc[(df['CNR'].isnull()) | ((df['CNR'].notnull()) & (df['DLY'] == 'Y'))] 
	df = df.drop(['CNR','CNL'], axis = 1)
	# 등록기호 랜덤으로 채우기
	REG_Range = ['SEw3NTk0', 'SEw3NzAz', 'SEw4MjM2', 'SEw4MDI4', 'SEw3NTE0', 'SEw4MDMx', 'SEw3NzU3', 'SEw3NTYw', 'SEw3NzY', 'SEw3NTA2', 'SEw3NTY4']
	REG_Randomset = []
	for i in range(229):
		REG_Randomset.append(rnd.choice(REG_Range))
	df['REG'].loc[df['REG'].isnull()] = REG_Randomset
	# 지연, 부정기편, 요일 수치화
	df['DLY'].loc[df['DLY'] == 'Y'] = 1
	df['DLY'].loc[df['DLY'] == 'N'] = 0
	df['IRR'].loc[df['IRR'] == 'Y'] = 1
	df['IRR'].loc[df['IRR'] == 'N'] = 0
	df['AOD'].loc[df['AOD'] == 'D'] = 1
	df['AOD'].loc[df['AOD'] == 'A'] = 0
	df['SDT_DY'].loc[df['SDT_DY'] == '월'] = 1
	df['SDT_DY'].loc[df['SDT_DY'] == '화'] = 2
	df['SDT_DY'].loc[df['SDT_DY'] == '수'] = 3
	df['SDT_DY'].loc[df['SDT_DY'] == '목'] = 4
	df['SDT_DY'].loc[df['SDT_DY'] == '금'] = 5
	df['SDT_DY'].loc[df['SDT_DY'] == '토'] = 6
	df['SDT_DY'].loc[df['SDT_DY'] == '일'] = 7
	# 시/분 단위로 나누기
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
	#DRR 지연사유 수치화
	df['DRR'].loc[df['DRR'].isnull()] = 0
	df['DRR'].loc[(
			(df["DRR"] == "C02") |
			(df["DRR"] == "C01") |
			(df["DRR"] == "D01") |
			(df["DRR"] == "C03") |
			(df["DRR"] == "C14") |
			(df["DRR"] == "B01")
	)] = 1
	df['DRR'].loc[(df['DRR'] != 0) & 
	(df['DRR'] != 1)] = 2

	## 나머지 수치화시키기
	label_encoder = preprocessing.LabelEncoder() 
	# AOD
	df_y = label_encoder.fit_transform(df['AOD']) 
	df['AOD'] = df_y.reshape(len(df_y), 1) 
	# REG
	df_y = label_encoder.fit_transform(df['REG']) 
	df['REG'] = df_y.reshape(len(df_y), 1)
	# FLO
	df_y = label_encoder.fit_transform(df['FLO']) 
	df['FLO'] = df_y.reshape(len(df_y), 1) 
	# ARP
	df_y = label_encoder.fit_transform(df['ARP']) 
	df['ARP'] = df_y.reshape(len(df_y), 1) 
	# ODP
	df_y = label_encoder.fit_transform(df['ODP']) 
	df['ODP'] = df_y.reshape(len(df_y), 1) 
	#쓸모없는 칼럼 삭제
	df = df.drop(['STT', 'ATT',"FLT"], axis = 1)

	return df