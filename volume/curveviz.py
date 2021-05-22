import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt 

df = pd.read_csv('prcurves.csv')
print(df)   

plt.plot(df.recall, df.precision, c='green')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prfig.png')