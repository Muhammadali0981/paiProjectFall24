import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 


# Outliers using box and whisker
df = pd.read_csv("ENB2012_data.csv")

col = df.columns
for c in col :
    df[c] = pd.to_numeric(df[c],errors='coerce')
    
df.fillna(-1)
df_cleaned = df.replace(-1, np.nan)

plt.figure(figsize=(16,12))

plt.subplot(5, 2, 1)
sns.boxplot(data = df, x="X1",legend=True)

plt.subplot(5, 2, 2)
sns.boxplot(data = df, x="X2",legend=True)

plt.subplot(5, 2, 3)
sns.boxplot(data = df, x="X3",legend=True)

plt.subplot(5, 2, 4)
sns.boxplot(data = df, x="X4",legend=True)

plt.subplot(5, 2, 5)
sns.boxplot(data = df, x="X5",legend=True)

plt.subplot(5, 2, 6)
sns.boxplot(data = df, x="X6",legend=True)

plt.subplot(5, 2, 7)
sns.boxplot(data = df, x="X7",legend=True)

plt.subplot(5, 2, 8)
sns.boxplot(data = df, x="X8",legend=True)

plt.subplot(5, 2, 9)
sns.boxplot(data = df, x="Y1",legend=True)

plt.subplot(5, 2, 10)
sns.boxplot(data = df, x="Y2",legend=True)
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()

col = df.columns
for c in col:
    sns.displot(df[c], kde = True, color = 'purple')
    plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr() ,annot = True, annot_kws= {"size": 8})
sns.set_palette('pastel')
plt.show()

plt.figure(figsize=(16,12))

plt.subplot(3, 2, 1)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["X1"], y=df["X2"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})


plt.subplot(3, 2, 2)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["X4"], y=df["X5"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})


plt.subplot(3, 2, 3)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["X4"], y=df["X2"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})


plt.subplot(3, 2, 4)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["Y1"], y=df["Y2"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})


plt.subplot(3, 2, 5)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["Y1"], y=df["X5"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})


plt.subplot(3, 2, 6)
sns.set_theme(style='darkgrid')
sns.regplot(x=df["Y2"], y=df["X5"], line_kws= {'color':'red'},scatter_kws ={'color':'darkblue', 'alpha':0.2, 'edgecolor':'black'})

plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.show()
