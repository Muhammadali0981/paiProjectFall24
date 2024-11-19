import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


df = pd.read_csv("ENB2012_data.csv")

col = df.columns
for c in col :
    df[c] = pd.to_numeric(df[c],errors='coerce')
    
df.fillna(-1, inplace=True)
df = df.replace(-1, np.nan)

colors_3 = ['#e74c3c', '#2980b9', '#27ae60']
colors_4 = ['#e74c3c', '#2980b9', '#27ae60', '#dc7633']

mean = df.mean()
plt.subplot(1, 3, 1)
plt.xlabel('data')
plt.ylabel('mean')
plt.bar(('X1', 'X7', 'X8'), (df['X1'].mean(), df['X7'].mean(), df['X8'].mean()), color = colors_3)


plt.subplot(1, 3, 2)
plt.xlabel('data')
plt.ylabel('mean')
plt.bar(('X2', 'X3', 'X4'), (df['X2'].mean(), df['X3'].mean(), df['X4'].mean()), color = colors_3)


plt.subplot(1, 3, 3)
plt.xlabel('data')
plt.ylabel('mean')
plt.bar(('X5', 'X6', 'Y1', 'Y2'), (df['X5'].mean(), df['X6'].mean(), df['Y1'].mean(), df['Y2'].mean()), color = colors_4)
plt.subplots_adjust(wspace=0.5)

plt.show()



std = df.std()
plt.subplot(1, 3, 1)
plt.xlabel('data')
plt.ylabel('std')
plt.bar(('X1', 'X7', 'X8'), (df['X1'].std(), df['X7'].std(), df['X8'].std()), color = colors_3)


plt.subplot(1, 3, 2)
plt.xlabel('data')
plt.ylabel('std')
plt.bar(('X2', 'X3', 'X4'), (df['X2'].std(), df['X3'].std(), df['X4'].std()), color = colors_3)


plt.subplot(1, 3, 3)
plt.xlabel('data')
plt.ylabel('std')
plt.bar(('X5', 'X6', 'Y1', 'Y2'), (df['X5'].std(), df['X6'].std(), df['Y1'].std(), df['Y2'].std()), color = colors_4)
plt.subplots_adjust(wspace=0.5)
plt.show()