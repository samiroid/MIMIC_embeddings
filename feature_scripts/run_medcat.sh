BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
NOTES_PATH=$BASE_PATH"DATA/input/all_full_notes.csv"
OUTPUT_PATH=$BASE_PATH"DATA/input/all_full_notes_ner.csv"
MEDCAT_VOCAB="/home/jainsarthak/vocab.dat"
MEDCAT_CDB="/home/jainsarthak/umls_base_wlink_fixed_x_avg_2m_mimic.dat"
                   
python transformers/nerl.py -input $NOTES_PATH \
                   -output $OUTPUT_PATH \
                   -medcat_vocab $MEDCAT_VOCAB \
                   -medcat_cdb $MEDCAT_CDB \
                   -ner \
                #    -build_tokenizer \