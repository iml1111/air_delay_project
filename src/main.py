from preprocess import *
from lgb import *
import pandas as pd

df = p_proc()
df2 = p_proc2()
df3 = label(pd.concat([df, df2], ignore_index=True))
#Y_pred = l_proc2(df3)
Y_pred, result = load_model(df3)
