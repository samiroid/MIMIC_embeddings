BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
NOTES_PATH=$BASE_PATH"DATA/input/mini_notes.csv"
OUTPUT_PATH=$BASE_PATH"DATA/input/"

python src/split_data.py -input $NOTES_PATH -output $OUTPUT_PATH 