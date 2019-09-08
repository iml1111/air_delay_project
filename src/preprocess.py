import numpy as np
import pandas as pd
import random as rnd
from sklearn import preprocessing

def p_proc(file = 'AFSNT.CSV'):
	###############파일 불러오기
	print("file_load...")
	df = pd.read_csv(file, engine='python', encoding="euc-kr")
	#############전처리하기
	print("pre-processing...")
	#결항 제거
	df = df.loc[df['CNL'] == 'N']
	#등록 기호 제거
	df = df[df['REG'].notnull()]
	#부정기편 제거하기
	df = df[ df['IRR'] == "N" ]
	# arp와 odp가 같을 경우 제거
	df = df[ df['ARP'] != df['ODP']  ]
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
	df_STT_Time = df['STT_H'] * 60 + df['STT_M']
	df_ATT_Time = df['ATT_H'] * 60 + df['ATT_M']
	df_H3 = df_ATT_Time - df_STT_Time
	# H3값이 음수면 빨리 출발, 양수면 늦은 출발(지연가능)
	df['Time'] = df_H3
	df = df.loc[(df['Time'] >= 0) | (df['DLY'] != 'Y') | (df['ATT_H'] < 0) | (df['ATT_H'] > 3) | (df['STT_H'] < 22)]
	df = df.drop(["REG",'IRR',"DRR","CNL","CNR",'ATT','Time',"ATT_H","ATT_M",], axis = 1)
	return df

def p_proc2(file = 'AFSNT_DLY.CSV'):
	df = pd.read_csv(file, engine='python', encoding="euc-kr")
	# 시/분 단위로 나누기
	STT_Hour = []
	STT_Minute = []
	STT = df['STT']
	sub = STT.str.split(':', expand = True)
	STT_Hour = sub.iloc[0:, 0]
	STT_Minute = sub.iloc[0:, 1]
	df['STT_H'] = STT_Hour.astype(int)
	df['STT_M'] = STT_Minute.astype(int)
	df = df.drop(["DLY_RATE"], axis = 1)
	return df

def label(df):
	df['DLY'].loc[df['DLY'] == 'Y'] = 1
	df['DLY'].loc[df['DLY'] == 'N'] = 0
	# ARP ODP 매핑
	df['ARP_ODP'] = df['ARP'] + df['ODP']
	arp = (df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').sum())/(df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').count())
	arp = arp[arp['DLY'] > 0.7]
	arp = arp.index
	for i in arp:
	    df['ARP_ODP'].loc[df['ARP_ODP']==i] = 1
	df['ARP_ODP'].loc[df['ARP_ODP']!=1] = 0
	# REG 다른 방식으로 추출
	df['SAME_DAY'] = (df['SDT_YY']*10000 + df['SDT_MM']*100 + df['SDT_DD']).astype(str) + df['FLO']
	df = df.drop(['ARP', 'ODP'], axis = 1)
	df['SDT_DY'].loc[df['SDT_DY'] == '월'] = 0
	df['SDT_DY'].loc[df['SDT_DY'] == '화'] = 1
	df['SDT_DY'].loc[df['SDT_DY'] == '수'] = 2
	df['SDT_DY'].loc[df['SDT_DY'] == '목'] = 3
	df['SDT_DY'].loc[df['SDT_DY'] == '금'] = 4
	df['SDT_DY'].loc[df['SDT_DY'] == '토'] = 5
	df['SDT_DY'].loc[df['SDT_DY'] == '일'] = 6
	label_encoder = preprocessing.LabelEncoder()
	df_y = label_encoder.fit_transform(df['FLO']) 
	df['FLO'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['SAME_DAY']) 
	df['SAME_DAY'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['AOD']) 
	df['AOD'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['FLT']) 
	df['FLT'] = df_y.reshape(len(df_y), 1)
	#학습 데이터 외의 칼럼 제거
	df = df.drop(['SDT_YY',"STT","STT_M"], axis = 1)
	return df
	# SDT_MM  SDT_DD  SDT_DY  FLO  FLT AOD  DLY  STT_H  ARP_ODP SAME_DAY