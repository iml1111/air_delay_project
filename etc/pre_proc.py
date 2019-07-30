import numpy as np # numpy 도 함께 import
import pandas as pd
print("loading...")
df = pd.read_csv('AFSNT.csv', encoding = "ISO-8859-1", engine='python')
# NaN -> "null"로 바꿔주기
print("preprocessing...")
df = df.fillna("null")
# 결항된 경우 날려버리기
df = df[df["CNL"] == "N"]
# 결항 관련 칼럼 날려버리기
df = df.loc[:,:"DRR"]
#C02,C01 만 남기고 다 제거하기
# (날씨, 검문등 예측불가변수 제외 )
#[970282 rows x 15 columns]
# df[(
# 	df["DLY"] == "N") | 
# 	((df["DLY"] == "Y") & (
# 							(df["DRR"] == "C02") |
# 							(df["DRR"] == "C01")
# 						)
# )]
# C02,C01,D01,C03,C14,B01 만 남기고 다 제거하기
# (날씨, 검문등 예측불가변수 제외 )
# (지연으로 인한 연쇄 작용 부분만 추려냈습니다)
# C02    108578  (AC정비) @
# C01      2042  (AC접속) @
# A01      1543  (안개)
# C10      1237  (제방빙작업)
# D01       957  (항로혼잡) @
# C03       913   (승객접속) @
# C14       879   (승무원연결) @
# Z99       669   (기타 가치 X)
# A05       608  (강풍)
# B01       418  (계류장혼잡) @
#[973429 rows x 15 columns]
df = df[(
	df["DLY"] == "N") | 
	((df["DLY"] == "Y") & (
							(df["DRR"] == "C02") |
							(df["DRR"] == "C01") |
							(df["DRR"] == "D01") |
							(df["DRR"] == "C03") |
							(df["DRR"] == "C14") |
							(df["DRR"] == "B01")
						)
)]
# DRR 칼럼(지연사유) 날려버리기
df = df.loc[:,:"DLY"]

# 실제시각-계획시간 칼럼 추가하기
# 각 데이터에 대하여 다음과 같은 연산을 시행
# 1. 깨진 요일을 되돌리기
# 2. 해당 시간을 분으로 표현하기
# 3. 지연여부를 고려하여 분(int) 끼리 값을 빼서 
# 실제시각과 계획시간의 차이를 계산하기
def time_int(x):
	a = x.split(":")[0]
	b = x.split(":")[1]
	return int(a)*60 + int(b)
def pre_func(x):
	if x['ATT'] == "null":
		x['ATT'] = x['STT']	
	if x["SDT_DY"] == 'ÀÏ':
		x["SDT_DY"] = "sun"
	elif x["SDT_DY"] == '¿ù':
		x["SDT_DY"] = "mon"
	elif x["SDT_DY"] == 'È­ ':
		x["SDT_DY"] = "tue"
	elif x["SDT_DY"] == '¼ö':
		x["SDT_DY"] = "wed"
	elif x["SDT_DY"] == '¸ñ':
		x["SDT_DY"] = "thu"
	elif x["SDT_DY"] == '±Ý':
		x["SDT_DY"] = "fri"
	elif x["SDT_DY"] == 'Åä':
		x["SDT_DY"] = "sat"
	x['ATT_int'] = time_int(x['ATT'])
	x['STT_int'] = time_int(x['STT'])
	ans =x['ATT_int'] - x['STT_int']
	if x['DLY'] == "N":
		if ans < 0:
			if ans + 1440 < 31:
				ans += 1440
		if ans >= 31: 
			ans -= 1440
	elif x['DLY'] == "Y":
		if ans < 31:
			ans += 1440
	x['TIME'] = ans
	return x

df = df.apply(lambda x: pre_func(x), axis = 1)
# 계획시간보다 너무 일찍 출발한 경우를 날려버리기
# -176을 경계로 매우 크게 차이남(데이터값 오류? 아니면 특례?)
df = df[(-173 <= df['TIME']) & ]

print("writing...")
# 해당 데이터를 csv로 생성하기
df.to_csv("pre_AFSNT.CSV", mode = "w")

#df[df['ATT'] == "null"]

