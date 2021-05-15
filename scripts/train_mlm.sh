BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"
DATASET="mini_full_notes"
DATASET="mini_notes"
CONF="test"

INPUT_PATH=$BASE_PATH"DATA/pretrain/"
OUTPUT_PATH=$BASE_PATH"DATA/output/"
TOK_PATH=$BASE_PATH"DATA/pretrain/tokenizer.json"
CONF_PATH=$BASE_PATH"DATA/confs/$CONF.json"
TRAINED_MODEL=$BASE_PATH"DATA/output/"$DATASET"_"$CONF".pt"
CHECKPOINT_PATH=$BASE_PATH"/DATA/pretrain/checkpt/"$DATASET"_"$CONF

python transformers/train_mlm.py -input $INPUT_PATH -output $OUTPUT_PATH \
                                -tok_path $TOK_PATH -conf_path $CONF_PATH \
                                -dataset $DATASET \
                                -checkpoint_path $CHECKPOINT_PATH \
                                -train \
                                -test \

                                # -load $TRAINED_MODEL \
                                