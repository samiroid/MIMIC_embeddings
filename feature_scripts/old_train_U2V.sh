U2V_PATH="/Users/samir/Dev/projects/U2V/U2V/u2v/"
BASE_PATH=" /Users/samir/Dev/projects/MIMIC_embeddings/MIMIC_embeddings/"

# U2V_PATH="/home/silvio/home/projects/U2V/u2v/"
# BASE_PATH="/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/"

CORPUS=$BASE_PATH"raw_data/sample.txt"
WORD_EMBEDDINGS=$BASE_PATH"DATA/features/word_embeddings.txt"
# PKL_PATH=$BASE_PATH"/DATA/pkl/"
OUTPUT_PATH=$BASE_PATH"DATA/embeddings/"

# python $U2V_PATH/build.py -input $CORPUS -emb $WORD_EMBEDDINGS -output $PKL_PATH

# python $U2V_PATH/train.py -input $PKL_PATH  -output $OUTPUT_PATH
python $U2V_PATH"u2v.py" -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -lr 0.01 \
                        -epochs 20 \
                        -neg_samples 1 \
                        -margin 5 \
                        -device cpu