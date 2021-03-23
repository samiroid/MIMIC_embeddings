#%%
from math import ceil
from scipy.sparse import lil
import streamlit as st
import pandas as pd
import seaborn as sns
import itertools
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from mimic.classify import *
import pickle 

SMALL_SIZE = 12
MEDIUM_SIZE = 17
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('text', usetex = True)


def plot_classification(df, ymin, ymax, tasks, groups):
        
    n_rows = ceil(len(tasks)/2)
    print(n_rows)
    coords = list(itertools.product(range(n_rows),range(n_rows)))       
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    n_cols = 2
    cols = st.beta_columns(n_cols)
    fdf = df[df["group"].isin(groups)]
    for t, c, i in zip(tasks, coords, range(len(tasks))):
        col = cols[i%n_cols]
        lil_df = fdf[fdf["task"] == t]        
        if len(lil_df) > 0:
            fig, ax = plt.subplots()
            lil_df.plot.bar(x="group",y="perf", title=t,color=cmap,legend=False,ax = ax)
            ax.set_ylim([ymin, ymax])
            col.pyplot(fig)
        else:
            print("task {} not found".format(t))

@st.cache(allow_output_mutation=True)
def plot_rocs(fname, title, output=None):
    with open(fname, "rb") as fi:
        rocs = pickle.load(fi)
    f,ax = plt.subplots(1,len(rocs),figsize=(5*len(rocs),5))
    for i, l in enumerate(rocs.keys()):    
        fpr_tprs, mean_fpr_tprs = rocs[l]
        plot_roc(fpr_tprs, mean_fpr_tprs,l,ax=ax[i])        
    plt.tight_layout()
    f.suptitle(title, y=1.02)
    return f
    # if output:
    #     plt.savefig(output,dpi=300, bbox_inches='tight')
    plt.show()
    


def plot_roc(fpr_tprs, mean_fpr_tprs, pos_class, ax=None, col=None,
                title=None, no_xticks=False, no_yticks=False ):
    color_dic = {"1":"steelblue", "0":"lightgray", "1M":"mediumblue", "0M":"gray"}          
    
    if not col:
        col = color_dic[pos_class]
    colm = color_dic[pos_class+"M"]
    if ax is None:
        fig,ax = plt.subplots(1,1)        
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k')    
    # st.write("ah fdss") 
    for i, [fpr, tpr] in enumerate(fpr_tprs):
        # st.write(i)
        ax.plot(fpr, tpr, lw=0.4,color=col)
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
    # if title is not None:
    #     ax.set_title(title)


def plot_similarity(tasks, model_name, data_path):
    for dataset in tasks:    
        fname = "{}rocs/rocs_{}_{}.pkl".format(data_path, model_name, dataset)
        print(dataset)
        print()        
        f = plot_rocs(fname,title=dataset)
        st.pyplot(f)

@st.cache(allow_output_mutation=True)
def read_results(data_path, model):
    df = pd.read_csv("{}{}".format(data_path, model))    
    df = df.drop_duplicates(subset=["task"],keep="last")    
    groups = df.columns[2:]
    df_long = pd.melt(df, id_vars=["model","task"], value_vars=groups, value_name="perf", var_name="group")
    return df_long



#%%
data_path = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/out/"
tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/tasks.txt"
demo_tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/demo_tasks.txt"

st.title("MIMIC")
task_type = st.sidebar.selectbox("Task Type", ["Clinical","Demographics"])
mini = st.sidebar.checkbox("mini", True)
if task_type == "Demographics":
    tasks = read_tasks(demo_tasks_fname, mini)
elif task_type == "Clinical":
    tasks = read_tasks(tasks_fname, mini)
# select_all = 
# select_none = 

selected_tasks = tasks[0]
if st.sidebar.button("all tasks"):
    selected_tasks = tasks
selected_tasks = st.sidebar.multiselect("Tasks", tasks, selected_tasks)

all_groups = ["all","men","women","white","black","hispanic","asian"]
# selected_groups = all_groups
# if st.sidebar.button("all groups"):
#     selected_groups = st.sidebar.multiselect("groups", groups, groups)
# selected_groups = st.sidebar.multiselect("groups", groups, groups)
# if st.sidebar.button("all groups"):


selected_groups = st.sidebar.multiselect("groups", all_groups, all_groups)
# if st.sidebar.button("all groups"):
#     selected_groups = st.sidebar.multiselect("groups", all_groups, all_groups)
# else:
#     selected_groups = st.sidebar.multiselect("groups", all_groups, [])





model = st.selectbox("Model", ["u2v","w2v"])
analysis = st.selectbox("Analysis", ["","probes", "similarity"])
if analysis == "probes":
    df = read_results(data_path, model)    
    ymin = st.sidebar.slider("ymin",min_value=0.0, max_value=0.5, value=0.3)
    ymax = st.sidebar.slider("ymax",min_value=0.5, max_value=1.1, value = 0.9)
    if ymax < ymin:
        st.error("max < min")
    plot_classification(df, ymin, ymax, selected_tasks, selected_groups)
elif analysis == "similarity":
    plot_similarity(selected_tasks, model, data_path)

# tasks = ["mini-AAURF", "mini-SIL","mini-ACD","cona"]
# all_tasks = list(set(df["task"]))







# tasks = {t:st.sidebar.checkbox(t) for t in all_tasks}
# checked_tasks = [t for t, c in tasks.items() if c]
# st.write(checked_tasks)
# plot_classification(df, tasks)



#%%



# # %%
# z = sns.barplot(x="task",y="perf", hue="group")

# # %%
# z = sns.barplot(x="grl",y="perf", hue="group")
# g = sns.FacetGrid(df_long, col="task", col_wrap=4, height=4, aspect=1, hue="group")
# g.map(z)


# %%
