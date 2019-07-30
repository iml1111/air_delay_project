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
	#분 영향 테스트
	def pre_time(x):
		hour = x['STT'].split
		############
	df = df.apply(lambda x: pre_time(x), axis = 1)

	#학습 데이터 외의 칼럼 제거
	df = df.drop(['CNL','CNR','IRR','DRR','SDT_YY','ATT'], axis = 1)

	

	# 모든 데이터 수치화
	# df['DLY'].loc[df['DLY'] == 'Y'] = 1
	# df['DLY'].loc[df['DLY'] == 'N'] = 0
	# df['SDT_DY'].loc[df['SDT_DY'] == '월'] = 0
	# df['SDT_DY'].loc[df['SDT_DY'] == '화'] = 1
	# df['SDT_DY'].loc[df['SDT_DY'] == '수'] = 2
	# df['SDT_DY'].loc[df['SDT_DY'] == '목'] = 3
	# df['SDT_DY'].loc[df['SDT_DY'] == '금'] = 4
	# df['SDT_DY'].loc[df['SDT_DY'] == '토'] = 5
	# df['SDT_DY'].loc[df['SDT_DY'] == '일'] = 6
	return df


df = proc()