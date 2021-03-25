import tadat.core as core
import pandas as pd
import pickle
import os
TMP_PATH = ""
CLINICALBERT = "emilyalsentzer/Bio_ClinicalBERT"

def read_cache(path):
    """ read a pickled object
        
        path: path
        returns: object
    """
    X = None
    try:
        with open(path, "rb") as fi:            
            X = pickle.load(fi)
    except FileNotFoundError:
        pass
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
        X = core.features.BOW(data, vocab_size,sparse=True)
    elif feature_type == "BOW-FREQ":
        X = core.features.BOW_freq(data, vocab_size,sparse=True)
    elif feature_type == "BOE-BIN":
        X = core.features.BOE(data, embeddings,"bin")
    elif feature_type == "BOE-SUM": 
        X = core.features.BOE(data, embeddings,"sum")
    elif feature_type == "U2V": 
        X = core.features.BOE(data, embeddings,"bin")
    elif feature_type == "BERT-POOL":
        X =  core.transformer_encoders.encode_sequences(data, batchsize=64)        
    elif feature_type == "BERT-CLS":
        X =  core.transformer_encoders.encode_sequences(data, cls_features=True,
                                                        batchsize=64)            
    elif feature_type == "MULTI-BERT-POOL":
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "MULTI-BERT-CLS":
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, 
                                                         cls_features=True,
                                                         batchsize=32,
                                                         tmp_path=TMP_PATH)
    elif feature_type == "CLINICALBERT-POOL":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_sequences(data, batchsize=64, tokenizer=tokenizer,
                                                                    encoder=encoder)        
    elif feature_type == "CLINICALBERT-CLS":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_sequences(data, cls_features=True,batchsize=64,
                                                                    tokenizer=tokenizer, encoder=encoder)        
    elif feature_type == "CLINICALMULTI-BERT-POOL":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, batchsize=32,tmp_path=TMP_PATH,
                                                              tokenizer=tokenizer, encoder=encoder)
    elif feature_type == "CLINICALMULTI-BERT-CLS":
        tokenizer, encoder = core.transformer_encoders.get_encoder(CLINICALBERT)
        X =  core.transformer_encoders.encode_multi_sequences(data, 10, cls_features=True, 
                                                                batchsize=32,tmp_path=TMP_PATH,
                                                                tokenizer=tokenizer, encoder=encoder)
    else:
        raise NotImplementedError
    return X

def extract_features(notes_path, feature_type, features_path):
    """ extract features and save features

        method will first look for computed features on disk and return them if found; otherwise, the features are computed and stored      
        
        notes_path: path to the clinical notes
        feature_type: type of feature (e.g bag of words, BERT)
        features_path: directory where the data can be found
                
        returns: list of subject ids and feature matrix -- the order of ids corresponds to order of the instances in the feature matrix
    """
    X = read_cache(features_path+"feats_{}".format(feature_type))
    if X:
        print("[reading cached features]")
        subject_ids, X_feats = X
    else:
        print("[computing {} features]".format(feature_type))
        df = pd.read_csv(notes_path, sep="\t", header=0)
        subject_ids = list(df["SUBJECT_ID"])
        docs = list(df["TEXT"])
        if "BERT" in feature_type:
            X_feats = get_features(docs, None, feature_type)
        elif "U2V" in feature_type:
            X, user_vocab = core.vectorizer.docs2idx([str(s) for s in subject_ids])
            user_embeddings = core.embeddings.read_embeddings(features_path+"/user_embeddings.txt", user_vocab)
            X_feats = get_features(X, len(user_vocab), feature_type, user_embeddings)
        else:
            embeddings = None
            X, word_vocab = core.vectorizer.docs2idx(docs)
            if "BOE" in feature_type:
                embeddings = core.embeddings.read_embeddings(features_path+"/word_embeddings.txt", word_vocab)
            X_feats = get_features(X, len(word_vocab), feature_type, embeddings)
        #save features
        print("[saving features]")
        write_cache(features_path+"feats_{}".format(feature_type), 
                    [subject_ids, X_feats])
    return subject_ids, X_feats
    