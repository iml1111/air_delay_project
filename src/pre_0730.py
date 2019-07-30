import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import random as rnd
from sklearn import preprocessing

def proc(filename = 'AFSNT.csv'):
	###############파일 불러오기
	print("file_load...")
	origin_df = pd.read_csv(filename, engine='python')
	df = pd.read_csv(filename, engine='python')
	#############전처리하기
	print("pre-processing...")
	#결항일 경우 지연 여부랑 사유도 똑같이 덮어쓰기
	temp = df['CNL'].loc[(df['CNL'] == 'Y')]
	df["DLY"].loc[(df['CNL'] == 'Y')] = temp
	temp = df['CNR'].loc[(df['CNL'] == 'Y')]
	df["DRR"].loc[(df['CNL'] == 'Y')] = temp
	#부정기편 제거하기
	df = df[ df['IRR'] == "N" ]
	#등록기호가 없을 경우 제거
	df = df[df['REG'].notnull()]
	# 실제시간이 없을 경우 제거
	df = df[df['ATT'].notnull()]
	#AC 관련 지연 사유 외에 모두 제거
	df = df[ (df["DRR"] == "C02") | (df["DRR"] == "C01") | (df['DRR'].isnull())]
	# REG 등록기호 빈도수가 상위 220개(약 1600번 이상) 이하 없애기
	temp = df.groupby("REG").size().sort_values(ascending = False).head(220)
	reg_list = list(temp.index)
	df['REG'] = df['REG'].apply(lambda x: x if x in reg_list else "null")
	df = df[df["REG"] != "null"]
	# FLT 편명 빈도수가 상위 650개 이하 날리기(500개)
	temp = df.groupby("FLT").size().sort_values(ascending = False).head(650)
	reg_list = list(temp.index)
	df['FLT'] = df['FLT'].apply(lambda x: x if x in reg_list else "null")
	df = df[df["FLT"] != "null"]
	#시간 int로 바꾸기
	def pre_time(x):
		hour = x['STT'].split(":")[0]
		min = x['STT'].split(":")[1]
		x['STT'] = int(hour)*60 + int(min)
		hour = x['ATT'].split(":")[0]
		min = x['ATT'].split(":")[1]
		x['ATT'] = int(hour)*60 + int(min)
		ans = x['ATT'] - x['STT']
		if x['DLY'] == "N":
			if ans < 0:
				if (ans + 1440 <= 30 and x['AOD'] == 'D') or (ans + 1440 <= 59 and x['AOD'] =='A'):
					ans += 1440
				if (ans >= 31 and x['AOD'] == 'D') or (ans >= 60 and x['AOD'] == 'A'): 
					ans -= 1440
		else:
			if (ans < 31 and x['AOD'] == 'D') or (ans < 60 and x['AOD'] == 'A'):
				ans += 1440
			if ans > 1471:
				ans -= 1440
		x['TIME'] = ans
		return x
	df = df.apply(lambda x: pre_time(x), axis = 1)
	df = df[(-173 <= df['TIME']) & (df['TIME'] <= 660)]

	#학습 데이터 외의 칼럼 제거
	df = df.drop(['CNL','CNR','IRR','DRR','SDT_YY','ATT','TIME'], axis = 1)

	# 모든 데이터 수치화
	# ARP, ODP int화 시키기(이거만 따로함)
	def pre_arp(x):
		x['ARP'] = int(x['ARP'].split("P")[1])
		x['ODP'] = int(x['ODP'].split("P")[1])
		return x
	df = df.apply(lambda x: pre_arp(x), axis = 1)
	df['DLY'].loc[df['DLY'] == 'Y'] = 1
	df['DLY'].loc[df['DLY'] == 'N'] = 0
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
	df_y = label_encoder.fit_transform(df['FLT']) 
	df['FLT'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['REG']) 
	df['REG'] = df_y.reshape(len(df_y), 1)
	df_y = label_encoder.fit_transform(df['AOD']) 
	df['AOD'] = df_y.reshape(len(df_y), 1)

	return df
