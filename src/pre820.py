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
	df = df.loc[(df['Time'] >= -180) | (df['DLY'] != 'Y') | (df['ATT_H'] < 0) | (df['ATT_H'] > 3) | (df['STT_H'] < 22)]
	df = df.loc[(df['STT_H'] != 0) & (df['STT_H'] != 1) & (df['STT_H'] != 23)]
	df['STT_H'].loc[(df['STT_H'] <= 6)] = 0
	df['STT_H'].loc[(df['STT_H'] >= 7) & (df['STT_H'] <= 12)] = 1
	df['STT_H'].loc[(df['STT_H'] >= 13) & (df['STT_H'] <= 19)] = 2
	df['STT_H'].loc[(df['STT_H'] == 20)] = 3
	df['STT_H'].loc[(df['STT_H'] >= 21) & (df['STT_H'] <= 23)] = 4
	return df

def label(df):
	#학습 데이터 외의 칼럼 제거
	df = df.drop(['CNL','CNR','IRR','DRR','SDT_YY','ATT','Time',"STT","ATT","ATT_H","ATT_M","STT_M"], axis = 1)
	df['DLY'].loc[df['DLY'] == 'Y'] = 1
	df['DLY'].loc[df['DLY'] == 'N'] = 0
	# ARP ODP 매핑
	sz = 15
	for i in range(1, sz+1):
	    df['ARP'].loc[df['ARP'] == ('ARP'+str(i))] = i
	    df['ODP'].loc[df['ODP'] == ('ARP'+str(i))] = i

	df['ARP_ODP'] = df['ARP'].astype(str) + df['ODP'].astype(str)
	arp = (df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').sum())/(df[['ARP_ODP', 'DLY']].groupby('ARP_ODP').count())
	arp = arp[arp['DLY']>0.1]
	arp = arp.index
	for i in arp:
	    df['ARP_ODP'].loc[df['ARP_ODP']==i] = 1
	df['ARP_ODP'].loc[df['ARP_ODP']!=1] = 0
	for i in arp:
	    df['ARP_ODP'].loc[df['ARP_ODP']==i] = 1
	df['ARP_ODP'].loc[df['ARP_ODP']!=1] = 0
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
	df_y = label_encoder.fit_transform(df['REG']) 
	df['REG'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['AOD']) 
	df['AOD'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['FLT']) 
	df['FLT'] = df_y.reshape(len(df_y), 1)
	return df
	# SDT_MM  SDT_DD  SDT_DY  FLO  FLT  REG  AOD  DLY  STT_H  ARP_ODP  ARP__ODP