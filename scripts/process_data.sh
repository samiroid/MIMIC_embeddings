BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
NOTES_PATH=$BASE_PATH"DATA/input/notes/filtered_notes.csv"
OUTPUT_PATH=$BASE_PATH"DATA/input/notes/"

python src/process_data.py -input $NOTES_PATH -output $OUTPUT_PATH -build_u2v