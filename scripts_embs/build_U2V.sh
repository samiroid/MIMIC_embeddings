
BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"

CONF_NAME=$1
OUTPUT_PATH=$BASE_PATH"/DATA/embeddings/u2v_debug/"
INPUT=$BASE_PATH"/DATA/input/notes/u2v_mini_filtered_notes.csv"
CONF=$BASE_PATH"/DATA/confs/u2v/"$CONF_NAME".json"
DEVICE="cuda:1"
# ENCODER_TYPE="bert"
# ENC_BATCH_SIZE=256
if [ -z $1 ]; then
        echo "Missing config file"
        exit 0
fi
echo "## BUIILDING WITH CONF " $CONF_NAME

python -m u2v.run -input $INPUT -output $OUTPUT_PATH \
                        -device $DEVICE \
                        -conf_path $CONF \
                        -build \
                        -encode 
