import tadat
import tadat.core 
from tadat.pipeline import plots
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc

from pdb import set_trace


SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex = True)

def plot_heatmap(data,labels,classes=None,legend=True,ax=None,title=None):    
    color_dic = {"depression":"steelblue", "control":"lightgray", "ptsd":"firebrick"}
    color_dic = {"1":"steelblue", "0":"lightgray", "3":"firebrick"}
    cols = {u:color_dic[c] for u,c in labels.items()}   
    # Creates a new figure
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(12,6))    
    cvr = colors.ColorConverter()
    tmp = sorted(cols.keys())
    cols_rgb = [cvr.to_rgb(cols[k]) for k in tmp]
    intervals = np.array(tmp + [tmp[-1]+1])
    cmap, norm = colors.from_levels_and_colors(intervals,cols_rgb)
    ax.pcolor(data,cmap = cmap, norm = norm)
    plt.xlim(0,data.shape[1]+5)
    plt.xticks([], [])

    max_y = data.shape[0]+17
    plt.ylim(0,max_y)
    #number of users per class
    slice_size=data.shape[1]/len(classes)

    for i,c in enumerate(classes):
        x=(slice_size*i)+(slice_size/2)
        ax.annotate(c,(x,data.shape[0]+5),color="k",size=10)

def plot_rocs(rocs, title, output=None):
    f,ax = plt.subplots(1,len(rocs),figsize=(10*len(rocs),10))
    for i, l in enumerate(rocs.keys()):    
        fpr_tprs, mean_fpr_tprs = rocs[l]
        plot_roc(fpr_tprs, mean_fpr_tprs,l,ax=ax[i])        
    plt.tight_layout()
    f.suptitle(title, y=1.02)
    if output:
        plt.savefig(output,dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc(fpr_tprs, mean_fpr_tprs, pos_class, ax=None, col=None,
                title=None, no_xticks=False, no_yticks=False ):
    color_dic = {"1":"steelblue", "0":"lightgray", "1M":"mediumblue", "0M":"gray"}                
    if not col:
        col = color_dic[pos_class]
    colm = color_dic[pos_class+"M"]
    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(10,10))    

    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
    for fpr, tpr in fpr_tprs:
        ax.plot(fpr, tpr, lw=0.1,color=col)
    mean_fpr, mean_tpr = mean_fpr_tprs
    au = auc(mean_fpr, mean_tpr)
    ax.plot(mean_fpr, mean_tpr, lw=3, color=colm,label=pos_class+ "\nAUC: %.2f" % au)
    if no_xticks: 
        ax.set_xticks([])
    else:
        ax.set_xticks([0.2,0.4,0.6,0.8])
    if no_yticks: 
        ax.set_yticks([])
    else:
        ax.set_yticks([0.2,0.4,0.6,0.8])
    ax.legend(loc='lower right', shadow=True)
    if title is not None:
        ax.set_title(title)