from preprocess import *
from lgb import *

df = p_proc()
df = label(df)
print(df)
Y_pred = l_proc(df)