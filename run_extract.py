# %%
%load_ext autoreload
%autoreload 2
from mimic.extract import *
mimic_path = "/Users/samir/Dev/resources/datasets/MIMIC/full/"
tasks_path = "/Users/samir/Dev/projects/MIMIC_embeddings/raw_data/"
out_data_path = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/"

# %%
# read tasks data
in_hospital_train = tasks_path+"in_hospital_train.csv"
in_hospital_test = tasks_path+"in_hospital_test.csv"
phenotyping_train = tasks_path+"phenotyping_train.csv"
phenotyping_test = tasks_path+"phenotyping_test.csv"
    
df_ihm_train = read_ihm(in_hospital_train)
df_ihm_test = read_ihm(in_hospital_test)
df_pheno_train = read_pheno(phenotyping_train)
df_pheno_test = read_pheno(phenotyping_test)
patient_ids = []
for df in [df_ihm_train, df_ihm_test, df_pheno_train, df_pheno_test]:
    patient_ids += df['SUBJECT_ID'].tolist()
patient_ids = list(set(patient_ids))
print("n patients: {}".format(len(patient_ids)))

# read and save demographics
df_patients = read_patients(mimic_path)
df_patients = df_patients.reset_index()
df_patients = df_patients[df_patients["SUBJECT_ID"].isin(patient_ids)]
# patients with demographics
patient_ids = list(set(df_patients["SUBJECT_ID"].tolist()))
print("n patients: {}".format(len(patient_ids)))

# %%
# read and save notes
if os.path.exists(out_data_path+"full_notes.csv"):
    df_notes = pd.read_csv(out_data_path+"full_notes.csv", sep="\t")
else:
    df_notes = read_notes(mimic_path, patient_ids)
    df_notes.to_csv(out_data_path+"full_notes.csv", index=False, sep="\t", header=True)
# patients with notes
patient_ids = list(set(df_notes["SUBJECT_ID"].tolist()))
print("n patients: {}".format(len(patient_ids)))

# %%
# save tasks data
ihm_ltr = len(df_ihm_train)
df_ihm_train2 = df_ihm_train[df_ihm_train["SUBJECT_ID"].isin(patient_ids)]
ihm_ltr2 = len(df_ihm_train2)
print("IHM Train {} > {}".format(ihm_ltr, ihm_ltr2))
df_ihm_train = df_ihm_train2

ihm_lts = len(df_ihm_test)
df_ihm_test2 = df_ihm_test[df_ihm_test["SUBJECT_ID"].isin(patient_ids)]
ihm_lts2 = len(df_ihm_test2)
print("IHM Test {} > {}".format(ihm_lts, ihm_lts2))
df_ihm_test = df_ihm_test2

pheno_ltr = len(df_pheno_train)
df_pheno_train2 = df_pheno_train[df_pheno_train["SUBJECT_ID"].isin(patient_ids)]
pheno_ltr2 = len(df_pheno_train2)
print("Phenotyping Train {} > {}".format(pheno_ltr, pheno_ltr2))
df_pheno_train = df_pheno_train2

pheno_lts = len(df_pheno_test)
df_pheno_test2 = df_pheno_test[df_pheno_test["SUBJECT_ID"].isin(patient_ids)]
pheno_lts2 = len(df_pheno_test2)
print("Phenotyping Test {} > {}".format(pheno_lts, pheno_lts2))
df_pheno_test = df_pheno_test2
df_patients = df_patients[df_patients["SUBJECT_ID"].isin(patient_ids)]
#add demographics
df_pheno_demo_test = df_pheno_test.set_index("SUBJECT_ID").join(df_patients.set_index("SUBJECT_ID"), how="inner", lsuffix="_").reset_index()
df_pheno_demo_train = df_pheno_train.set_index("SUBJECT_ID").join(df_patients.set_index("SUBJECT_ID"), how="inner", lsuffix="_").reset_index()

# %%
save_tasks(df_ihm_train, df_ihm_test, df_pheno_demo_train, df_pheno_demo_test, out_data_path+"tasks/")
df_patients.to_csv(out_data_path+"demographics.csv", index=False, sep="\t", header=True)


# %%
df_notes
note_lens = [len(d.split()) for d in df_notes["TEXT"]]
note_lens

# %%
# pd.DataFrame(note_lens).plot.hist(bins=20)
# pd.DataFrame(note_lens,dtype=pd.Int32Dtype).plot.hist(bins=20)
from matplotlib import pyplot as plt
import numpy as np
plt.hist(note_lens, bins=1000, range=(0,10000))
min(note_lens)
np.mean(note_lens)
np.median(note_lens)
# %%
