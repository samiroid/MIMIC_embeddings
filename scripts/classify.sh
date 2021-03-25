MIMIC_PATH="/Users/samir/Dev/resources/datasets/MIMIC/full/"

BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings/"
TASKS_PATH=$BASE_PATH"raw_data/"
OUTPUT_PATH=$BASE_PATH"DATA/input/"


DATA_PATH = $BASE_PATH"/DATA/"

# %%
feature_type = "U2V"
metric = "auroc"
dataset = "mini-IHM"
tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/tasks.txt"
demo_tasks_fname = "/Users/samir/Dev/projects/MIMIC_embeddings/DATA/input/tasks/demo_tasks.txt"

python src/extract.py -mimic $MIMIC_PATH -data $TASKS_PATH -output $OUTPUT_PATH