
BASE_PATH="/Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings"
BASE_PATH="/home/s.amir/projects/MIMIC_embeddings"

CONF_NAME=$1
MASTER_CONFS=$BASE_PATH"/confs/u2v/master/"$CONF_NAME".json"
OUTPUT_PATH=$BASE_PATH"/confs/u2v/train/"$CONF_NAME
# CONF=$BASE_PATH"/confs/u2v/build/"$CONF_NAME".json"
# DEVICE="cuda:0"

if [ -z $1 ]; then
        echo "Missing config file"
        exit 0
fi
echo "## EXPLODING MASTER CONF" $CONF_NAME

python src/expand_confs.py -input $MASTER_CONFS -output $OUTPUT_PATH