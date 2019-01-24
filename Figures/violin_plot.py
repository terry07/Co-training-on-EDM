import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# provide path
#path = ...

#os.chdir()
df = pd.read_csv('for_violin_figure.csv')

sns.set(style="whitegrid")

plt.figure(figsize=(10, 5))
ax = sns.violinplot(x="alg", y="score", hue="metric", data = df, palette="muted", split = True, inner = 'quartile')

plt.legend(loc='lower left')
plt.xticks(rotation=15)   

ax.set_xlabel("",fontsize=10)
ax.set_ylabel("Score (%)",fontsize=10)
fig = ax.get_figure()
fig.savefig('f1_acc_proposed_against_full_Supervised.png', format='png', dpi=800)