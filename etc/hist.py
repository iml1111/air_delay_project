import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("pre_AFSNT.CSV")
COL = "FLO"
VAL = ["A","B","C","D","E","F","G","H","I","J","K","L"]
a,b = 5,5

for idx, val_ in enumerate(VAL):
	plt.subplot(a,b, idx + 1)
	plt.title( COL + "==" + str(val_))
	xs, ys, patches = plt.hist(
	df[df[COL] == val_]['TIME'], bins = 200,range=[-175,200])
	for i in range(len(patches)):
		if ys[i] <= 30:
			patches[i].set_facecolor('b')
		else:
			patches[i].set_facecolor('r')
plt.show()


#exec(open("test.py").read())
#SDT_YY  SDT_MM  SDT_DD SDT_DY   ARP   ODP  ... IRR   STT   ATT DLY ATT_int STT_int TIME
