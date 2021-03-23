# %%
import pandas as pd
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
import os 

from tadat.core import data

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.remove("no")
STOP_WORDS.remove("not")

def clean_text(txt):
    return " \t ".join([preprocess(t) for t in sent_tokenize(txt)]) 
    
def ethnicity_multi_labels(x):
    if "ASIAN" in x:
        return "ASIAN"
    elif "AFRICAN" in x:
        return "BLACK"
    elif "HISPANIC" in x:
        return "HISPANIC"
    elif "WHITE" in x:
        return "WHITE"
    elif "NATIVE" in x:
        return "OTHER"        
    else:
        return "OTHER"

def ethnicity_binary_labels(x):
    if "ASIAN" in x or "AFRICAN" in x or "HISPANIC" in x or "NATIVE" in x: 
        return "NON-WHITE"
    elif "WHITE" in x:
        return "WHITE"
    else:
        return "OTHER"

def get_patient_id(x):
    return int(x[:x.index("_")])

def preprocess(txt):
    txt = txt.lower().replace("\n"," ").replace("\t"," ")
    txt = re.sub("[0-9]+","*NUM*", txt)
    txt = " ".join([w for w in word_tokenize(txt) if w not in STOP_WORDS and len(w)>1])
    txt = txt.replace("***NUM*-*NUM*-*NUM***", "*DATE*").replace("***NUM*-*NUM***","*DATE*").replace("*NUM*:*NUM*","*TIME*")         
    return txt

def read_ihm(path):
    df = pd.read_csv(path)
    df['SUBJECT_ID'] = df['stay'].apply(get_patient_id) 
    df = df[["SUBJECT_ID","y_true"]]
    df = df.rename({'y_true': 'Y'}, axis=1)
    return df

def read_pheno(path):
    df = pd.read_csv(path)
    df['SUBJECT_ID'] = df['stay'].apply(get_patient_id)     
    return df

def read_notes(path, patient_ids):
    df_notes = pd.read_csv(path+"NOTEEVENTS.CSV.gz")
    df_notes["SUBJECT_ID"] = pd.to_numeric(df_notes["SUBJECT_ID"])
    df_notes = df_notes.set_index("SUBJECT_ID")
    #filter notes by category
    df_notes = df_notes[df_notes["CATEGORY"].isin(["Physician ","Nursing","Nursing/other"])]
    #filter by patient ids
    df_notes = df_notes[df_notes.index.isin(patient_ids)]   
    df_notes = df_notes.sort_values(by=["SUBJECT_ID","CHARTTIME"], ascending=False)
    df_notes = df_notes.reset_index()[["SUBJECT_ID","TEXT"]]
    df_notes["TEXT"] = df_notes["TEXT"].apply(clean_text)
    # df_notes["len"] = df_notes["TEXT"].apply(lambda x:len(x))
    # df_notes = df_notes[df_notes["len"] > 0]
    df_notes = df_notes.groupby(["SUBJECT_ID"], as_index = False).agg({'TEXT': ' [SEP] '.join})
    return df_notes

def read_patients(mimic_path):
    all_patients = pd.read_csv(mimic_path+"PATIENTS.CSV.gz")
    all_admissions = pd.read_csv(mimic_path+"ADMISSIONS.CSV.gz")
    all_admissions["SUBJECT_ID"] = pd.to_numeric(all_admissions["SUBJECT_ID"])
    all_patients["SUBJECT_ID"] = pd.to_numeric(all_patients["SUBJECT_ID"])
    all_patients = all_patients.set_index("SUBJECT_ID")
    all_admissions = all_admissions.set_index("SUBJECT_ID")
    #join dataframes
    patients = all_patients.join(all_admissions, how="inner", on=["SUBJECT_ID"],lsuffix="A_")
    patients["ETHNICITY_BINARY"] = patients["ETHNICITY"].apply(lambda x:ethnicity_binary_labels(x))
    patients["ETHNICITY"] = patients["ETHNICITY"].apply(lambda x:ethnicity_multi_labels(x))
    #exclude patients with ethnicity "other"
    patients = patients[patients["ETHNICITY"] != "OTHER"]
    #filter relevant columns
    patients = patients.reset_index()
    patients = patients.drop_duplicates(subset=["SUBJECT_ID"])
    patients = patients[["SUBJECT_ID","GENDER","INSURANCE","ETHNICITY","ETHNICITY_BINARY"]]
    return patients.set_index("SUBJECT_ID")

def split_data(df, y_label, split=0.8):
    #split into training and test sets
    train_split, test_split = data.shuffle_split_idx(df[y_label], split)
    df_train = df.iloc[train_split, :]
    df_test = df.iloc[test_split, :]
    return df_train, df_test

def save_dataset(df_train, df_test, name, path):    
    SAMPLE=1000
    df_train, df_val = split_data(df_train,"Y")
    df_train.to_csv(path+"{}_train.csv".format(name), index=False, sep="\t", header=True)    
    df_test.to_csv(path+"{}_test.csv".format(name), index=False, sep="\t", header=True)    
    df_val.to_csv(path+"{}_val.csv".format(name), index=False, sep="\t", header=True)    
    
    df_train.head(SAMPLE).to_csv(path+"mini-{}_train.csv".format(name),index=False, sep="\t", header=True)    
    df_test.head(SAMPLE).to_csv(path+"mini-{}_test.csv".format(name),index=False, sep="\t", header=True)    
    df_val.head(SAMPLE).to_csv(path+"mini-{}_val.csv".format(name),index=False, sep="\t", header=True)    

def save_tasks(df_ihm_train, df_ihm_test, df_pheno_train, df_pheno_test,
                 out_data_path):
    dirname = os.path.dirname(out_data_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    #save IHM
    save_dataset(df_ihm_train, df_ihm_test, "IHM", out_data_path)
    #save Phenotyping tasks
    disease_names = list(df_pheno_train.iloc[:,3:-4].columns)    
    abvs = set()
    dataset_names = ["IHM, In Hospital Mortality"]
    for d in disease_names:
        #convert disease name to initials 
        did = "".join([w[0] for w in d.upper().split()])    
        #remove non-alphanum characters
        did = re.sub(r'\W+', '', did)
        #avoid clashing names
        seqnum=2
        tdid=did
        while tdid in abvs:
            tdid=did+"-"+str(seqnum)
            seqnum+=1
        did=tdid
        #make sure that names have not clashed
        assert did not in abvs
        abvs.add(did)

        df_train = df_pheno_train[["SUBJECT_ID", d]]
        df_train = df_train.rename({d : 'Y'}, axis=1)
        df_test = df_pheno_test[["SUBJECT_ID", d]]
        df_test = df_test.rename({d : 'Y'}, axis=1)
        # train_prev = len(df_train[df_train["Y"]==1])/len(df_train)

        dataset_names.append("{},{}".format(did,d))
        # print("{} & {} & ${}$ \\\\".format(did,d, round(train_prev,2)))
        save_dataset(df_train, df_test, did, out_data_path)
    
    demo_tasks = ["GENDER", "ETHNICITY","ETHNICITY_BINARY"]
    for d in demo_tasks:
        df_train = df_pheno_train[["SUBJECT_ID", d]]
        df_train = df_train.rename({d : 'Y'}, axis=1)
        df_test = df_pheno_test[["SUBJECT_ID", d]]
        df_test = df_test.rename({d : 'Y'}, axis=1)        
        save_dataset(df_train, df_test, d, out_data_path)



    #save file with all tasks
    with open(out_data_path+"tasks.txt", "w") as fod:
        fod.write("\n".join(dataset_names))
    with open(out_data_path+"demo_tasks.txt", "w") as fod:
        dn = ["{},{}".format(d,d.lower()) for d in demo_tasks]
        # from pdb import set_trace; set_trace()
        fod.write("\n".join(dn))
    #split tasks into multiple files
    n_task_files = 2
    slice_size = int(len(dataset_names)/n_task_files)+1
    for i in range(n_task_files):        
        with open(out_data_path+"tasks_{}.txt".format(i+1), "w") as fod:
            fod.write("\n".join(dataset_names[i*slice_size:(i+1)*slice_size]))
