BASE_PATH="/Users/samir/Dev/projects/MIMIC/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"

INPUT=$BASE_PATH"/DATA/input/notes/filtered_notes.csv"
OUTPUT_PATH=$BASE_PATH"/DATA/probes/pkl/"
EMB_PATH=$BASE_PATH"/DATA/embeddings/user_embeddings.txt"
FEATURES="BOW-BIN"
python src/encode.py -input $INPUT -output $OUTPUT_PATH -embeddings $EMB_PATH \
                     -feature $FEATURES