# %%
%load_ext autoreload
%autoreload 2
import tadat
import tadat.core 
from tadat.pipeline import plots
from matplotlib import colors
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pdb import set_trace
import pickle
from mimic.classify import read_tasks
import os

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

def compute_rocs(user_ranking, user_labels, sim_scores):
    label_set = set(user_labels.values())
    res = {}
    for l in label_set:
        res[l] = compute_roc(user_ranking, user_labels, sim_scores, l)
    return res

def compute_roc(user_ranking, user_labels, sim_scores, pos_class):
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    n_users=0
    fpr_tprs = []
    for i in range(user_ranking.shape[1]):    
        rank = user_ranking[:,i]
        sim = sim_scores[:,i]
        user_idx = rank[0] 
        user_label = user_labels[user_idx]
        others_labels = [1 if user_labels[int(x)] == user_label else 0 for x in rank ]
        # set_trace()
        if user_label != pos_class:
            continue
        else:
            n_users+=1
        fpr, tpr, thresholds = roc_curve(others_labels, sim, drop_intermediate=True)
        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        fpr_tprs.append([fpr, tpr])        
    mean_tpr /= n_users
    mean_tpr[-1] = 1.0    
    return fpr_tprs, [mean_fpr, mean_tpr]

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

def run_dataset(dataset, model_name, U_path, data_path, output_path):
    df = pd.read_csv("{}{}_train.csv".format(data_path, dataset), sep="\t")
    df = df.drop_duplicates(subset=['SUBJECT_ID'])
    #find minority class and number of the corresponding number users
    max_users = df.groupby("Y").size().min()
    min_class = df.groupby("Y").size().argmin()
    min_class_users = df[df["Y"] == min_class]
    other_users = df[df["Y"] != min_class].iloc[:max_users, :]
    #all users
    users = min_class_users.append(other_users)
    user_vocab = {str(u):i for i,u in enumerate(users["SUBJECT_ID"])}
    user_labels = {i: str(label) for i, label in enumerate(users["Y"])}
    user_idxs = list(user_vocab.values())
    U = tadat.core.embeddings.read_embeddings(U_path, user_vocab)    
    ranking, scores = tadat.core.embeddings.similarity_rank(U,user_idxs)
    res = compute_rocs(ranking, user_labels, scores)
    fname = "{}/rocs_{}_{}.pkl".format(output_path, model_name, dataset)
    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(fname,"wb") as fod:
        pickle.dump(res, fod, -1)



#%%
#COMPUTE ROCS
model_name="U2V"
U_path = "DATA/features/user_embeddings.txt"
output_path = "DATA/out/rocs/"
data_path= "DATA/input/tasks/"
tasks  = read_tasks(data_path+"tasks.txt", True)
for dataset in tasks:
    print(dataset)
    run_dataset(dataset, model_name, U_path, data_path, output_path)



#%%
#PLOT ROCS
model_name="U2V"
output_path = "DATA/out/rocs/"
data_path= "DATA/input/tasks/"
tasks  = read_tasks(data_path+"tasks.txt", True)
for dataset in tasks:    
    fname = "{}/rocs_{}_{}.pkl".format(output_path, model_name, dataset)
    print(dataset)
    print()
    with open(fname, "rb") as fi:
        res = pickle.load(fi)
        plot_rocs(res,title=dataset)
    

#%%
# dataset="DATA/input/tasks/mini-DOLM_train.csv"
df = pd.read_csv(dataset, sep="\t")
df = df.drop_duplicates(subset=['SUBJECT_ID'])
#find minority class and number of the corresponding number users
max_users = df.groupby("Y").size().min()
min_class = df.groupby("Y").size().argmin()
min_class_users = df[df["Y"] == min_class]
other_users = df[df["Y"] != min_class].iloc[:max_users, :]
#all users
users = min_class_users.append(other_users)
user_vocab = {str(u):i for i,u in enumerate(users["SUBJECT_ID"])}
user_labels = {i: str(label) for i, label in enumerate(users["Y"])}
user_idxs = list(user_vocab.values())
U_path = "DATA/features/user_embeddings.txt"
U = tadat.core.embeddings.read_embeddings(U_path, user_vocab)
# 
# user_labels


#%%
output="DATA/out/rocs_u2v_dolm.png"
ranking, scores = tadat.core.embeddings.similarity_rank(U,user_idxs)
res = compute_rocs(ranking, user_labels, scores)
plot_rocs(res,title="DOLM",output=output)


# %%
# def roc_ranking(user_ranking, user_labels, sim_scores, col=None,
#                 title=None, no_xticks=False, no_yticks=False, output=None):
#     label_set = set(user_labels.values())
#     f,ax = plt.subplots(1,len(label_set),figsize=(10*len(label_set),10))
#     results  = {}
#     for i,l in enumerate(label_set):
#         auc = __roc_ranking(user_ranking, user_labels, sim_scores, pos_class=l, ax=ax[i], 
#                     col=col, title="", no_xticks=no_xticks, no_yticks=no_yticks)
#         results[l] = auc
#     plt.tight_layout()
#     f.suptitle(title, y=1.02)
#     if output:
#         plt.savefig(output,dpi=300, bbox_inches='tight')

# def __roc_ranking(user_ranking, user_labels, sim_scores, pos_class, ax=None, col=None,
#                 title=None, no_xticks=False, no_yticks=False):
    
#     # color_dic = {1:"steelblue", 0:"lightgray", 2:"firebrick"}    
#     color_dic = {"1":"steelblue", "0":"lightgray", "1M":"mediumblue", "0M":"gray"}    
#     # color_dic = {"1M":"slateblue", "0M":"gray"}    
#     if not col:
#         col = color_dic[pos_class]
#     colm = color_dic[pos_class+"M"]
#     if ax is None:
#         fig,ax = plt.subplots(1,1,figsize=(10,10))    
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')
#     mean_tpr = 0.0
#     mean_fpr = np.linspace(0, 1, 100)
#     n_users=0
#     for i in range(user_ranking.shape[1]):    
#         rank = user_ranking[:,i]
#         sim = sim_scores[:,i]
#         user_idx = rank[0] 
#         user_label = user_labels[user_idx]
#         others_labels = [1 if user_labels[int(x)] == user_label else 0 for x in rank ]
#         # set_trace()
#         if user_label != pos_class:
#             continue
#         else:
#             n_users+=1
#         fpr, tpr, thresholds = roc_curve(others_labels, sim, drop_intermediate=True)
#         mean_tpr += np.interp(mean_fpr, fpr, tpr)
#         mean_tpr[0] = 0.0
#         ax.plot(fpr, tpr, lw=0.1,color=col)
#     # set_trace()
#     mean_tpr /= n_users
#     mean_tpr[-1] = 1.0
#     au = auc(mean_fpr, mean_tpr)
#     ax.plot(mean_fpr, mean_tpr, lw=3, color=colm,label=pos_class+ "\nAUC: %.2f" % au)
#     if no_xticks: 
#         ax.set_xticks([])
#     else:
#         ax.set_xticks([0.2,0.4,0.6,0.8])
#     if no_yticks: 
#         ax.set_yticks([])
#     else:
#         ax.set_yticks([0.2,0.4,0.6,0.8])
#     ax.legend(loc='lower right', shadow=True)
#     if title is not None:
#         ax.set_title(title)
#     return au