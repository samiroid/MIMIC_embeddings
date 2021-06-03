
BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"
DATA_PATH=$BASE_PATH"/DATA/probes"
OUTPUT_PATH=$BASE_PATH"/DATA/probes/output/"
FEATURES="U2V_user_embeddings"
FEATURES="BOW-BIN"
# DATA_PATH = $BASE_PATH"/DATA/"

# %%
# feature_type = "U2V"
# metric = "auroc"
# dataset = "mini-IHM"
# tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/probes/tasks/tasks.txt"
# demo_tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/probes/tasks/demo_tasks.txt"

python src/classify.py -data $DATA_PATH -features $FEATURES -output $OUTPUT_PATH