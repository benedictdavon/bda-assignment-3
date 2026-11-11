"""Static figures for the Markdown dataset card; source tables remain inspectable."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'reports/data_audit'
d=pd.read_csv(out/'target_distribution.csv')
fig,ax=plt.subplots(figsize=(8,4))
bars=ax.bar(d.quality,d['count'],color='#315d8a')
ax.bar_label(bars,labels=[f'{n:,}' for n in d['count']],padding=4,fontsize=10)
ax.set(xlabel='Observed quality score',ylabel='Training rows',title='Training labels: 82.44% have quality 5 or 6',ylim=(0,7100),xticks=range(3,9))
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(out/'target_distribution.png',dpi=150);plt.close(fig)
d=pd.read_csv(out/'drift_and_outliers.csv').sort_values('target_spearman')
fig,ax=plt.subplots(figsize=(8,5))
ax.barh(d.feature,d.target_spearman,color='#315d8a')
ax.axvline(0,color='black',lw=.6)
ax.set(xlabel='Spearman correlation with quality',title='Marginal associations in the synthetic training data',xlim=(-.15,.45))
ax.spines[['top','right']].set_visible(False)
fig.tight_layout();fig.savefig(out/'target_associations.png',dpi=150);plt.close(fig)
print('Wrote two static audit figures from recorded CSV tables.')
