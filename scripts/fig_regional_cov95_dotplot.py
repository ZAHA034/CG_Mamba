"""Regional Cov95 dot plot (replaces the 6-panel choropleth). Reads 3 canonical sources:
 - cg_mamba: runs/apmd_diagnostic/apmd_residuals.csv (per-region Cov95, h1-4 avg)
 - lstm/patchtst/vanilla_mamba: runs/phase_3_region_wis.csv (tS_cov95_h1..4)
 - dlinear/epideep: runs/phase_3_region_wis_extras.csv
All means verified to match the manuscript (CG 0.954, PatchTST 0.695, Vanilla 0.571, LSTM 0.513, EpiDeep 0.382, DLinear 0.286).
"""
import pandas as pd, numpy as np
import matplotlib as mpl; mpl.use("Agg")
import matplotlib.pyplot as plt
mpl.rcParams.update({"font.family":"serif","font.serif":["STIXGeneral","DejaVu Serif"],
 "mathtext.fontset":"stix","pdf.fonttype":42,"ps.fonttype":42,"font.size":7,
 "axes.linewidth":0.6,"axes.edgecolor":"#4D4D4D","xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6})
regs=[f"hhs{i}" for i in range(1,11)]
def cov_wis(f,models):
    e=pd.read_csv(f); cc=['tS_cov95_h1','tS_cov95_h2','tS_cov95_h3','tS_cov95_h4']
    e['cov']=e[cc].mean(axis=1); t=e.groupby(['baseline','region'])['cov'].mean().unstack('region')
    return {m:t.loc[m] for m in models if m in t.index}
data={}
data.update(cov_wis('runs/phase_3_region_wis.csv',['lstm','patchtst','vanilla_mamba']))
data.update(cov_wis('runs/phase_3_region_wis_extras.csv',['dlinear_ensemble_gauss','epideep']))
d=pd.read_csv('runs/apmd_diagnostic/apmd_residuals.csv'); r=d[d.scope!='national'].copy()
r['cov']=(np.abs(r.resid)<=1.96*np.sqrt(r.s2_total)).astype(float)
data['cg_mamba']=r.groupby('scope')['cov'].mean()
# order + style (Okabe-Ito), CG emphasized
spec=[('cg_mamba','CG-Mamba','#0072B2','o',34,1.0),('patchtst','PatchTST','#E69F00','s',16,0.9),
 ('vanilla_mamba','Vanilla Mamba','#009E73','^',16,0.9),('lstm','LSTM','#D55E00','v',16,0.9),
 ('epideep','EpiDeep','#56B4E9','D',13,0.9),('dlinear_ensemble_gauss','DLinear-ens','#CC79A7','P',16,0.9)]
fig,ax=plt.subplots(figsize=(3.45,2.9),layout="constrained")
ax.axvline(0.95,color="#444444",lw=0.8,ls="--",zorder=1)
ax.text(0.95,9.7,"nominal 0.95",fontsize=5.5,color="#444444",ha="center",va="bottom")
yy=np.arange(10)
for key,lab,c,mk,ms,al in spec:
    s=data[key].reindex(regs)
    ax.scatter(s.values,yy,marker=mk,s=ms,c=c,edgecolors='black' if key=='cg_mamba' else 'none',
               linewidths=0.4,alpha=al,zorder=5 if key=='cg_mamba' else 3,label=lab)
ax.set_yticks(yy); ax.set_yticklabels([f"HHS{i}" for i in range(1,11)])
ax.set_xlim(0.25,1.03); ax.set_xlabel("95% prediction-interval coverage (Cov95, $h{=}1$–4 mean)")
ax.set_ylim(-0.6,9.9); ax.grid(True,axis='x',color="#DDDDDD",lw=0.4)
ax.legend(loc="lower left",ncol=2,framealpha=0.92,handletextpad=0.2,columnspacing=0.8,borderpad=0.3)
out="CGM_v2_paper/figures/regional_cov95_dotplot"
fig.savefig(out+".pdf"); fig.savefig(out+".png",dpi=300)
cg=data['cg_mamba']
print(f"CG mean {cg.mean():.3f} SD {cg.std():.3f} min {cg.min():.3f} max {cg.max():.3f}")
print("wrote",out+".pdf")
