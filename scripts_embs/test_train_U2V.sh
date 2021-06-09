
BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"

CONF_NAME=$1
OUTPUT_PATH=$BASE_PATH"/DATA/embeddings/u2v_test/"
# NOTES=$BASE_PATH"/DATA/input/notes/u2v_mini_filtered_notes.csv"
CONF=$BASE_PATH"/confs/u2v/train/"$CONF_NAME".json"
DEVICE="cuda:1"
# ENCODER_TYPE="bert"
# ENC_BATCH_SIZE=256
if [ -z $1 ]; then
        echo "Missing config file"
        exit 0
fi
echo "## TRAINING WITH CONF " $CONF_NAME

python -m u2v.run -output $OUTPUT_PATH \
                  -device $DEVICE \
                  -conf_path $CONF \
                  -train 
                  
                
