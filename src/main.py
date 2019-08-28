from pre828 import *
from lgb import *

df = p_proc()
df = label(df)
Y_pred = l_proc(df)