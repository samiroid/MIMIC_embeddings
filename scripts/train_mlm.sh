BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
INPUT_PATH=$BASE_PATH"DATA/pretrain/"
OUTPUT_PATH=$BASE_PATH"DATA/output/"
TOK_PATH=$BASE_PATH"DATA/pretrain/tokenizer.json"
CONF_PATH=$BASE_PATH"DATA/confs/test.json"
DATASET="mini_full_notes"
TRAINED_MODEL=$BASE_PATH"DATA/output/model_mini_full_notes_test.pt"
                   
python transformers/train_mlm.py -input $INPUT_PATH -output $OUTPUT_PATH \
                                -tok_path $TOK_PATH -conf_path $CONF_PATH \
                                -dataset $DATASET \
                                -load $TRAINED_MODEL \
                                -test
                                # -train \
                                