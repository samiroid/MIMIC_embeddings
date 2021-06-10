#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:v100-sxm2:1
#SBATCH --output=/home/s.amir/projects/MIMIC_embeddings/DATA/slurm_logs/build_U2V_out_%j.txt
#SBATCH --error=/home/s.amir/projects/MIMIC_embeddings/DATA/slurm_logs/build_U2V_err_%j.txt
#SBATCH --mem 60G
#SBATCH --time 08:00:00
module load gcc
BASE_PATH="/home/s.amir/projects/MIMIC_embeddings"

source $BASE_PATH"/env/bin/activate"
pip install -e /home/s.amir/projects/U2V/
pip install -r $BASE_PATH"/requirements.txt"

STEP=$1
CONF_NAME=$2
OUTPUT_PATH=$BASE_PATH"/DATA/embeddings/u2v/"
NOTES=$BASE_PATH"/DATA/input/notes/u2v_filtered_notes.csv"
CONF=$BASE_PATH"/confs/u2v/build/"$CONF_NAME".json"
DEVICE="auto"

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

