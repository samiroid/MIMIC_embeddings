
BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"

EMBEDDINGS="/home/silvio/home/resources/BioWord2vec200.txt" 
EMBEDDINGS="/home/silvio/home/resources/BioWordVec_PubMed_MIMICIII_d200.bin" 

OUTPUT_PATH=$BASE_PATH"/DATA/embeddings/u2v_debug/"
INPUT=$BASE_PATH"/DATA/input/notes/u2v_filtered_notes.csv"

ENCODER_TYPE="fasttext"
DEVICE="cuda:1"
ENCODER_TYPE="bert"
ENC_BATCH_SIZE=256

BUILD=1
TRAIN=0
if (($BUILD > 0)); then
    python -m u2v.run -input $INPUT -emb $EMBEDDINGS -output $OUTPUT_PATH \
                            -min_word_freq 1 \
                            -min_docs_user 1 \
                            -encoder_type $ENCODER_TYPE \
                            -encoder_batchsize $ENC_BATCH_SIZE \
                            -device $DEVICE \
                            -build \
                            -max_docs_user 10 \
                            -encode 
fi

# if (($TRAIN > 0)); then
#     python -m u2v.run -input $INPUT -emb $EMBEDDINGS -output $OUTPUT_PATH \
#                             -lr 1 \
#                             -epochs 20 \
#                             -margin 5 \
#                             -val_split 0.2 \
#                             -batch_size 128 \
#                             -device $DEVICE \
#                             -encoder_type $ENCODER_TYPE \
#                             -train
# fi
