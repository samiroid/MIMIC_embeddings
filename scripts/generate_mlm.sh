BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
NOTES_PATH=$BASE_PATH"DATA/input/"
FEATS_PATH=$BASE_PATH"DATA/pretrain/"
TOK_PATH=$BASE_PATH"DATA/pretrain/tokenizer.json"
DATASET="mini_full_notes"
                   
python transformers/generate_mlm_data.py -input $NOTES_PATH -output $FEATS_PATH \
                   -vectorize \
                   -create_mlm \
                   -build_tokenizer \
                   -dataset $DATASET \
                  #  -tok_path $TOK_PATH \