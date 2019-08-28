from pre828 import *
from le828 import *

df_train = p3_proc()
df_test = pd.read_csv("AFSNT_DLY.CSV", engine='python', encoding="euc-kr")
df_test = df_test.drop(["DLY_RATE"], axis = 1)

df_train = label(df_train)
df_test = label2(df_test)

df = pd.concat([df_train, df_test])
test_proc(df)