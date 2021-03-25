
import tadat
import tadat.core 
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd
from pdb import set_trace
import pickle
import os


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