import pre_0730
from learn_0730 import *
import pandas as pd
df = pd.read_csv("pre_0730.csv")

def col_test():
	c_list = ["SDT_MM", 
				"SDT_DD", 
				"SDT_DY", 
				"ARP" ,
				"ODP" ,
				"FLO" ,
				"FLT" ,
				"REG" ,
				"AOD" ,
				"STT"]
	for i in c_list:
		proc_rf(df.loc[:,[i,"DLY"]],i,500)