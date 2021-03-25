import argparse
import pandas as pd
import os
from sklearn.linear_model import SGDClassifier 
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import precision_recall_fscore_support as score
from tadat import core
from .vectorize import extract_features


GROUPS = { "GENDER": ["M","F"],   
         "ETHNICITY": ["WHITE","BLACK","ASIAN","HISPANIC"]
}

def train_classifier(X_train, Y_train, X_val, Y_val, 
                     init_seed, shuffle_seed=None, input_dimension=None):    
    """ train a classifier
        X_train: training instances 
        Y_yrain: training labels
        X_val: validation instances
        Y_val: validation labels
        init_seed: parameter initialization seed
        shuffle_seed: data shuffling seed
        input_dimension: number of input features
        
        returns: fitted classifier
    """
    x = SGDClassifier(loss="log", random_state=init_seed)
    x.fit(X_train, Y_train)
    
    return x

def evaluate_classifier(model, X_test, Y_test,
                   labels, pos_label, model_name, random_seed, subgroup, res_path=None):
    """ evaluate a classifier
        model: classifier to be evaluated        
        X_test: test instances
        Y_test: test labels
        labels: label set
        model_name: model name
        random_seed: random seed that was used to train the classifier
        subgroup: demographic subgroup represented in the data
        res_path: path to save the results
        
        returns: dictionary of evaluation wrt to different metrics
    """
    Y_hat = model.predict(X_test)
    Y_hat_prob = model.predict_proba(X_test)
    #get probabilities for the positive class
    # if CLASSIFIER == 'sklearn':
    # from pdb import set_trace; set_trace()
    Y_hat_prob = Y_hat_prob[:,labels[pos_label]]    
    microF1 = f1_score(Y_test, Y_hat, average="micro") 
    macroF1 = f1_score(Y_test, Y_hat, average="macro") 
    try:
        aurocc = roc_auc_score(Y_test, Y_hat_prob)
    except ValueError:
        aurocc = 0
    try:
        prec, rec, thresholds = precision_recall_curve(Y_test, Y_hat_prob)       
        auprc = auc(rec, prec)
    except ValueError:
        auprc = 0
    try:
        tn, fp, fn, tp = confusion_matrix(Y_test, Y_hat).ravel()
        specificity = tn / (tn+fp)
        sensitivity = tp / (fn+tp)
    except ValueError:
        specificity, sensitivity = 0, 0
    
    res = {"model":model_name, 
            "seed":random_seed,  
            "group":subgroup,    
            "microF1":round(microF1,3),
            "macroF1":round(macroF1,3),
            "auroc":round(aurocc,3),
            "auprc":round(auprc,3),
            "specificity":round(specificity,3),
            "sensitivity":round(sensitivity,3)           
            }

    if res_path is not None:    
        core.helpers.save_results(res, res_path, sep="\t")
    return res

def vectorize(df_train, df_val, df_test, subject_ids):
    """ vectorize the instances and stratify them by demographic subgroup
        df_train: training data as a DataFrame
        df_test: test data as a DataFrame
        df_val: validation data as a DataFrame
        subject_ids: list of subject ids (the order corresponds to order of the features that were extracted)
        
        returns: vectorized train, validation and test datasets, stratified by demographic subgroup
                 label vocabulary                 
    """

    #vectorize labels
    train_Y = df_train["Y"].tolist()
    val_Y = df_val["Y"].tolist()           
    test_Y = df_test["Y"].tolist()               
    label_vocab = core.vectorizer.get_labels_vocab(train_Y+val_Y)    
    train_Y,_ = core.vectorizer.label2idx(train_Y, label_vocab)
    val_Y,_ = core.vectorizer.label2idx(val_Y, label_vocab)
    test_Y,_ = core.vectorizer.label2idx(test_Y, label_vocab)      
    
    #get indices into the feature matrix
    train_idxs = [subject_ids.index(i) for i in list(df_train["SUBJECT_ID"])] 
    val_idxs = [subject_ids.index(i) for i in list(df_val["SUBJECT_ID"])] 
    test_idxs = [subject_ids.index(i) for i in list(df_test["SUBJECT_ID"])] 
    #construct datasets
    train = {}
    test = {}
    val = {}
    #unstratified 
    train["all"] = [train_idxs, train_Y]
    test["all"] = [test_idxs, test_Y]
    val["all"] = [val_idxs, val_Y]
    #stratified by demographics 
    for group in list(GROUPS.keys()):
        #and subgroups
        for subgroup in GROUPS[group]:                
            df_train_sub = df_train[df_train[group] == subgroup]
            df_test_sub = df_test[df_test[group] == subgroup]
            df_val_sub = df_val[df_val[group] == subgroup]
            #vectorize labels               
            train_Y_sub,_ = core.vectorizer.label2idx(df_train_sub["Y"], label_vocab)            
            test_Y_sub,_ = core.vectorizer.label2idx(df_test_sub["Y"], label_vocab)            
            val_Y_sub,_ = core.vectorizer.label2idx(df_val_sub["Y"], label_vocab)      
            #get indices into the feature matrix
            train_idxs_sub = [subject_ids.index(i) for i in list(df_train_sub["SUBJECT_ID"])] 
            test_idxs_sub = [subject_ids.index(i) for i in list(df_test_sub["SUBJECT_ID"])] 
            val_idxs_sub = [subject_ids.index(i) for i in list(df_val_sub["SUBJECT_ID"])] 
            if subgroup == "M":
                subgroup = "men"
            elif subgroup == "F":
                subgroup = "women"
            train[subgroup.lower()] = [train_idxs_sub, train_Y_sub]
            test[subgroup.lower()] = [test_idxs_sub, test_Y_sub]
            val[subgroup.lower()] = [val_idxs_sub, val_Y_sub]

    return train, val, test, label_vocab

def read_dataset(path, dataset_name, df_demographics):    
    
    """ read dataset        
        path: path to the dataset
        dataset_name: name of the dataset
        df_demographics: DataFrame of patients
                
        returns: train, test and validation sets as DataFrames
    """
    df_train = pd.read_csv("{}/{}_train.csv".format(path, dataset_name), 
                           sep="\t", header=0)
    df_test  = pd.read_csv("{}/{}_test.csv".format(path, dataset_name),
                           sep="\t", header=0)
    df_val   = pd.read_csv("{}/{}_val.csv".format(path, dataset_name),
                           sep="\t", header=0)
    #set indices
    df_demographics.set_index("SUBJECT_ID", inplace=True)
    df_train.set_index("SUBJECT_ID", inplace=True)
    df_test.set_index("SUBJECT_ID", inplace=True)
    df_val.set_index("SUBJECT_ID", inplace=True)

    df_train = df_train.join(df_demographics, on="SUBJECT_ID", 
                             how="inner", lsuffix="N_").reset_index()
    df_test = df_test.join(df_demographics, on="SUBJECT_ID", 
                           how="inner", lsuffix="N_").reset_index()
    df_val = df_val.join(df_demographics, on="SUBJECT_ID", 
                         how="inner", lsuffix="N_").reset_index()

    return df_train, df_test, df_val   

def read_tasks(tasks_fname, mini=False):
    datasets = []
    with open(tasks_fname,"r") as fid:                
        for i,l in enumerate(fid):            
            task_abv, task_name = l.strip("\n").split(",")
            dataset = "mini-"+task_abv if mini else task_abv
            datasets.append(dataset)
    return datasets

def run(data_path, dataset, feature_type, metric, pos_label=1):    
    #read patients data
    df_demographics = pd.read_csv(data_path+"/input/demographics.csv", 
                              sep="\t", header=0)
    
    #read dataset
    df_train, df_test, df_val = read_dataset(data_path+"/input/tasks/", dataset, df_demographics)
    
    print("[{} > train/test set size: {}/{}]".format(dataset, len(df_train), len(df_test)))
    
    #extract features
    subject_ids, feature_matrix = extract_features(data_path+"/input/full_notes.csv", 
                                                    feature_type, data_path+"/features/")      
    train, val, test, label_vocab = vectorize(df_train, df_val, df_test, subject_ids)
    inv_label_vocab = core.vectorizer.invert_idx(label_vocab)
    train_idx, train_Y = train["all"]
    val_idx, val_Y = val["all"]
    #slice the feature matrix to get the corresponding instances
    train_X = feature_matrix[train_idx, :]    
    val_X = feature_matrix[val_idx, :]        
        
    groups = list(val.keys())    
    init_seed = 1
    shuffle_seed = 2
    seed = "{}x{}".format(init_seed, shuffle_seed)
    model = train_classifier(train_X, train_Y,val_X, val_Y,  
                                input_dimension=train_X.shape[-1],
                                init_seed=init_seed, 
                                shuffle_seed=shuffle_seed)                                                                                
    #test each subgroup (note thtat *all* is also a subgroup)
    results = {"model":feature_type, "task":dataset, "metric":metric}
    for subgroup in groups:                                
        test_idx_sub, test_Y_sub = test[subgroup]                 
        test_X_sub = feature_matrix[test_idx_sub, :]                
        res_sub = evaluate_classifier(model, test_X_sub, test_Y_sub, 
                                    label_vocab, pos_label, feature_type, seed, subgroup)                
        results[subgroup]= res_sub[metric]     
    #save results
    dirname = os.path.dirname(data_path+"/out/")
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    
    df_results = pd.DataFrame.from_records([results]) 
    if not os.path.exists(data_path+"/out/"+dataset):
        df_results.to_csv(data_path+"/out/"+dataset, index=False, header=True)
    else:
        df_results.to_csv(data_path+"/out/"+dataset, index=False, mode="a", header=False)
    
    if not os.path.exists(data_path+"/out/"+feature_type.lower()):
        df_results.to_csv(data_path+"/out/"+feature_type.lower(), index=False, header=True)
    else:
        df_results.to_csv(data_path+"/out/"+feature_type.lower(), index=False, mode="a", header=False)

    return df_results

def cmdline_args():
    parser = argparse.ArgumentParser(description="Extract MIMIC data ")
    parser.add_argument('-tasks', type=str, required=True, help='path to tasks data')    
    parser.add_argument('-data', type=str, required=True, help='path to data')       
    parser.add_argument('-metric', type=str, required=True, help='metric')    
    parser.add_argument('-feature', type=str, required=True, help='feature type')    
    parser.add_argument('-probe', choices=["clinical", "demographics"], default="clinical", 
                        help='feature type')    
        
    return parser.parse_args()	

def clinical_probe(data_path, tasks_path, feature_type, metric):
    
    tasks  = read_tasks(tasks_path, True)
    for dataset in tasks:
        run(data_path, dataset, feature_type, metric)

def demographics_probe(data_path, tasks_path, feature_type, metric):
    tasks  = read_tasks(tasks_path, True)
    metric = "macroF1"
    for dataset, pos_label in zip(tasks, ["M","WHITE","WHITE"]):
        run(data_path, dataset, feature_type, metric, pos_label=pos_label)
    

if __name__ == "__main__":
    args = cmdline_args()
    if args.probe == "clinical":
        clinical_probe(args.data, args.tasks, args.feature, args.metric)
    elif args.probe == "demographics":
        demographics_probe(args.data, args.tasks, args.feature, args.metric)
    else:
        NotImplementedError

    