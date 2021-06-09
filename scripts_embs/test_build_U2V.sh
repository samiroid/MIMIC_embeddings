BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"

STEP=$1
CONF_NAME=$2
OUTPUT_PATH=$BASE_PATH"/DATA/embeddings/u2v_test/"
NOTES=$BASE_PATH"/DATA/input/notes/u2v_mini_filtered_notes.csv"
CONF=$BASE_PATH"/confs/u2v/test/"$CONF_NAME".json"
DEVICE="cuda:0"
BUILD="build"
if [ -z $1 ]; then
        echo "Missing step"
        exit 0
fi
if [ -z $2 ]; then
        echo "Missing config file"
        exit 0
fi


if [ "$STEP" = "build" ]; then
        echo "## BUIILDING WITH CONF " $CONF_NAME
        python -m u2v.run -docs $NOTES -output $OUTPUT_PATH \
                                -device $DEVICE \
                                -conf_path $CONF \
                                -build            

fi

if [ "$STEP" = "encode" ]; then
        echo "## ENCODING WITH CONF " $CONF_NAME
        python -m u2v.run -docs $NOTES -output $OUTPUT_PATH \
                        -device $DEVICE \
                        -conf_path $CONF \
                        -encode \
                        -cache
                        
fi


