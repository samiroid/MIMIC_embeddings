from tadat.core import vectorizer
from tadat.core import features as featurizer
from tadat.core import embeddings as embedding_utils
from tadat.core import transformer_encoders

import pandas as pd
import pickle
import os
import argparse

TMP_PATH = ""
CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"

def read_cache(path):
    """ read a pickled object
        
        path: path
        returns: object
    """
    X = None
    print(f"read {path}")
    try:
        with open(path, "rb") as fi:            
            X = pickle.load(fi)
    except FileNotFoundError:
        print(f"file {path} not found")
    return X

def write_cache(path, o):
    """ pickle an object
            
        path: path
        o: object
    """
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(path, "wb") as fo:
        pickle.dump(o, fo)


def get_features(data, vocab_size, feature_type, embeddings=None):
    """ compute features from the data
        data: data instances
        vocab_size: size of the vocabulary
        feature_type: type of feature (e.g bag of words, BERT)
        word_vectors: path to pretrained (static) word vectors
        
        returns: feature matrix
    """
    if feature_type == "BOW-BIN":
        X = featurizer.BOW(data, vocab_size,sparse=True)
    elif feature_type == "BOW-FREQ":
        X = featurizer.BOW_freq(data, vocab_size,sparse=True)
    elif feature_type == "BOE-BIN":
        X = featurizer.BOE(data, embeddings,"bin")
    elif feature_type == "BOE-SUM": 
        X = featurizer.BOE(data, embeddings,"sum")
    elif feature_type == "U2V": 
        X = featurizer.BOE(data, embeddings,"bin")
    elif feature_type == "BERT-POOL":
        X =  transformer_encoders.encode_sequences(data, batchsize=64)        
    elif feature_type == "BERT-CLS":
        X =  transformer_encoders.encode_sequences(data, cls_features=True,
                                                        batchsize=64)            
    elif feature_type == "MULTI-BERT-POOL":
        X =  transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "MULTI-BERT-CLS":
        X =  transformer_encoders.encode_multi_sequences(data, 10, 
                                                         cls_features=True,
                                                         batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "CLINICALBERT-POOL":
        tokenizer, encoder = transformer_encoders.get_encoder(CLINICALBERT)
        X =  transformer_encoders.encode_sequences(data, batchsize=64, tokenizer=tokenizer,
                                                                    encoder=encoder)        
    elif feature_type == "CLINICALBERT-CLS":
        tokenizer, encoder = transformer_encoders.get_encoder(CLINICALBERT)
        X =  transformer_encoders.encode_sequences(data, cls_features=True,batchsize=64,
                                                                    tokenizer=tokenizer, encoder=encoder)        
    elif feature_type == "CLINICALMULTI-BERT-POOL":
        tokenizer, encoder = transformer_encoders.get_encoder(CLINICALBERT)
        X =  transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,tmp_path=TMP_PATH,
                                                              tokenizer=tokenizer, encoder=encoder)
    elif feature_type == "CLINICALMULTI-BERT-CLS":
        tokenizer, encoder = transformer_encoders.get_encoder(CLINICALBERT)
        X =  transformer_encoders.encode_multi_sequences(data, 10, cls_features=True, 
                                                                batchsize=32,tmp_path=TMP_PATH,
                                                                tokenizer=tokenizer, encoder=encoder)
    else:
        raise NotImplementedError
    return X

def extract_features(input_path, output_path, feature_type, embeddings_path="", 
                    pretrained_weights=""):
    """ extract features and save features

        method will first look for computed features on disk and return them if found; otherwise, the features are computed and stored      
        
        input_path: path to the clinical notes
        feature_type: type of feature (e.g bag of words, BERT)
        output_path: directory where the data can be found
                
        returns: list of subject ids and feature matrix -- the order of ids corresponds to order of the instances in the feature matrix
    """
    # X = read_cache(output_path+"feats_{}".format(feature_type))
    # if X:
    #     print("[reading cached features]")
    #     subject_ids, X_feats = X
    # else:
    print("[computing {} features]".format(feature_type))
    
    # fname =  output_path+"feats_{}".format(feature_type)
    fname = f"{output_path}{feature_type}" 

    df = pd.read_csv(input_path, sep="\t", header=0)
    subject_ids = list(df["SUBJECT_ID"])
    docs = list(df["TEXT"])
    if "BERT" in feature_type:
        X_feats = get_features(docs, None, feature_type)
    elif "U2V" in feature_type:
        assert embeddings_path
        fname+="_"+os.path.basename(os.path.splitext(embeddings_path)[0])
        X, user_vocab = vectorizer.docs2idx([str(s) for s in subject_ids])
        user_embeddings = embedding_utils.read_embeddings(embeddings_path, user_vocab)
        X_feats = get_features(X, len(user_vocab), feature_type, user_embeddings)
    else:
        embeddings = None
        X, word_vocab = vectorizer.docs2idx(docs)
        if "BOE" in feature_type:
            embeddings = embedding_utils.read_embeddings(embeddings_path, word_vocab)
        X_feats = get_features(X, len(word_vocab), feature_type, embeddings)
    #save features
    print(f"[saving features @ {fname}]")
    write_cache(fname, 
                [subject_ids, X_feats])
    return subject_ids, X_feats

def cmdline_args():
    parser = argparse.ArgumentParser(description="Vectorize MIMIC data ")    
    parser.add_argument('-input', type=str, required=True, help='path to data')       
    parser.add_argument('-output', type=str, required=True, help='path to data')       
    parser.add_argument('-feature', type=str, required=True, help='feature type')    
    parser.add_argument('-embeddings', type=str, help='embeddings path')    
    parser.add_argument('-pretrained', type=str, help='pretrained weights')    
            
    return parser.parse_args()	

if __name__ == "__main__":
    args = cmdline_args()
    extract_features(args.input, args.output, args.feature, args.embeddings, 
                     args.pretrained)    
