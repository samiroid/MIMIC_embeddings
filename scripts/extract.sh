MIMIC_PATH="/Users/samir/Dev/resources/datasets/MIMIC/full/"

BASE_PATH="/Users/samir/Dev/projects/MIMIC/MIMIC_embeddings/MIMIC_embeddings/"

TASKS_PATH=$BASE_PATH"raw_data/"
OUTPUT_PATH=$BASE_PATH"DATA/input/"

python src/extract.py -mimic $MIMIC_PATH -data $TASKS_PATH -output $OUTPUT_PATH