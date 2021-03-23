# %%
%load_ext autoreload
%autoreload 2
from mimic.classify import *
data_path = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/"

# %%
feature_type = "U2V"
metric = "auroc"
dataset = "mini-IHM"
tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/tasks.txt"
tasks  = read_tasks(tasks_fname, True)
for dataset in tasks:
    run(data_path, dataset, feature_type, metric)

# %%
demo_tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/demo_tasks.txt"
demo_tasks  = read_tasks(demo_tasks_fname, True)
metric = "macroF1"
for dataset, pos_label in zip(demo_tasks, ["M","WHITE","WHITE"]):
    run(data_path, dataset, feature_type, metric, pos_label=pos_label)


# %%

# %%
