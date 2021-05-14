BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
NOTES_PATH=$BASE_PATH"DATA/input/"
FEATS_PATH=$BASE_PATH"DATA/pretrain/"
TOK_PATH=$BASE_PATH"DATA/pretrain/tokenizer.json"

                   
python transformers/generate_mlm_data.py -input $NOTES_PATH -output $FEATS_PATH \
                   -tok_path $TOK_PATH \
                   -vectorize \
                #    -create_mlm \
                #    -build_tokenizer \