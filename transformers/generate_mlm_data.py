import argparse
import math
from sys import set_asyncgen_hooks
import torch
from tokenizers import Tokenizer
import time
from ipdb import set_trace
import pandas as pd
from random import random, randrange, randint, shuffle, choice
import collections
import pandas as pd
import numpy as np 
import ast

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace

import os
MAX_PREDS=20
MAX_SEQ_LEN=512
MIN_SEQ_LEN=20

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])

#adapted from https://github.com/MLforHealth/HurtfulWords
def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq,  vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    ignore = set(["[CLS]", "[SEP]", "TIME","DATE","DOCTOR"])
    for (i, token) in enumerate(tokens):
        #do not mask non alphanumeric and special tokens
        if token in ignore or not token.isalpha():
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq,
                      max(1, int(round(len(tokens) * masked_lm_prob))))
    shuffle(cand_indices)
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(
                index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def get_mlm(dataset, tokenizah, max_seq_len, max_preds=20, mask_prob=0.15):
    #unpack list of keys from vocabulary (dict)
    vocab = [*tokenizah.get_vocab()]
    instances = []
    for x in dataset:
        notes = x.split("[EON]")
        #try to pack sentences together
        sent_accum = ["[CLS]"]
        last_sent = None
        for note in notes:                        
            sentences = note.split("[EOS]")            
            for s in sentences:
                tokens = tokenizah.encode(s+" [SEP] ").tokens
                last_sent = tokens
                sent_accum += last_sent
                if len(sent_accum) >= max_seq_len:
                    #truncate
                    sent_accum = sent_accum[:max_seq_len-1] + ["[SEP]"]
                    # set_trace()
                    z = create_masked_lm_predictions(sent_accum, mask_prob, max_preds, vocab)                
                    masked_tokens, masked_indices, masked_token_labels = z

                    x = {"tokens":tokens,
                    "masked_tokens":masked_tokens,
                    "masked_token_labels":masked_token_labels,
                    "masked_indices":masked_indices,
                    "len":len(masked_tokens)}
                    instances.append(x)
                    #add last sentence to the beguining of next sequence
                    sent_accum = ["[CLS]"]+last_sent
        #last sequence of the patient
        #only save sequences longer than MIN_SEQ_LEN
        if len(sent_accum) >= MIN_SEQ_LEN:
            z = create_masked_lm_predictions(sent_accum, mask_prob, max_preds, vocab)                
            masked_tokens, masked_indices, masked_token_labels = z
            x = {"tokens":tokens,
            "masked_tokens":masked_tokens,
            "masked_token_labels":masked_token_labels,
            "masked_indices":masked_indices,
            "len":len(masked_tokens)}
            instances.append(x)
        else:
            print(f"skipped seq size {len(sent_accum)}")    
    return instances
    

def create_mlm_data(input_path, output_path, dataset, tokenizah, max_seq_len, max_preds=20, 
                    mask_prob=0.15, test_split=0.2, val_split=0.2, anno=True ):   
    
    df = pd.read_csv(input_path, sep="\t")
    N = len(df)
    test_idx=math.ceil(N*test_split)
    val_idx=math.ceil(N*val_split)
    df_test = df.iloc[:test_idx]
    df_val = df.iloc[test_idx:(test_idx+val_idx)]
    df_train = df.iloc[(test_idx+val_idx):]
    if anno:
        train_docs = df_train["ANNO_TEXT"]
        test_docs = df_test["ANNO_TEXT"]
        val_docs = df_val["ANNO_TEXT"]
    else:
        train_docs = df_train["TEXT"]
        test_docs = df_test["TEXT"]
        val_docs = df_val["TEXT"]
    print("train: ", len(train_docs))
    print("test: ", len(test_docs))
    print("val: ", len(val_docs))
    
    for data, sp in zip([train_docs, test_docs, val_docs], ["train","test","val"]):
        instances = get_mlm(data, tokenizah, max_seq_len, max_preds, mask_prob)
        df = pd.DataFrame(instances)
        df.to_csv(f"{output_path}{dataset}_{sp}.csv")

def feats(row, max_seq_len, max_preds, tokenizah):
    tokens = ast.literal_eval(row["masked_tokens"])
    token_label = ast.literal_eval(row["masked_token_labels"])
    masked_indices = ast.literal_eval(row["masked_indices"])
    #truncate sequences longer than max_seq_len
    tokens = tokens[:max_seq_len]
    #remove masked indices that may have been truncated
    masked_indices = [mi for mi in masked_indices if mi <= max_seq_len]
    #remove the corresponding labels
    token_label = token_label[:len(masked_indices)]        
    #convert tokens to ids
    token_ids = [tokenizah.token_to_id(t) for t in tokens]
    label_ids = [tokenizah.token_to_id(t) for t in token_label]    
    #add padding to tokens and labels
    tokens_pad = [0] * (max_seq_len - row["len"])
    labels_pad = [0] * (max_preds - len(label_ids))        
    token_ids += tokens_pad    
    label_ids += labels_pad    
    #masked indices have the same size as the labels so we use the same padding
    masked_indices += labels_pad
    try:
        assert len(label_ids) == max_preds
    except AssertionError:
        set_trace()
    try:
        assert len(token_ids) == max_seq_len
    except AssertionError:
        set_trace()
    token_ids = np.array(token_ids).reshape(1,-1)    
    label_ids = np.array(label_ids).reshape(1,-1)    
    #build mask for padding positions
    attention_mask = np.zeros_like(token_ids)
    # attention_mask[:,masked_indices] = 0
    attention_mask[:, row["len"]:] = 1

    return token_ids, label_ids, attention_mask, masked_indices

def vectorize(path, dataset, tokenizah, max_seq_len, max_preds):    
    splits = ["train","val","test"]
    fpath = "{}{}_{}.csv"    
    df = pd.read_csv(fpath.format(path, dataset, "train"))
    # max_seq_len = df["len"].max()
    for sp in splits:
        df = pd.read_csv(fpath.format(path, dataset, sp))
        tokens = []
        labels = []
        attention_masks = []    
        masked_indices = []    
        for i in range(len(df)):
            z = feats(df.iloc[i,:], max_seq_len, max_preds, tokenizah)
            tok,lab,attn,x =  z
            tokens.append(tok)
            labels.append(lab)
            attention_masks.append(attn)       
            masked_indices.append(x) 
        tokens = np.vstack(tokens)
        labels = np.vstack(labels)
        attention_masks = np.vstack(attention_masks)
        masked_indices = np.vstack(masked_indices)
        
        np.save(f"{path}{dataset}_{sp}_tokens",tokens)
        np.save(f"{path}{dataset}_{sp}_labels",labels)
        np.save(f"{path}{dataset}_{sp}_mask",attention_masks)
        np.save(f"{path}{dataset}_{sp}_x",masked_indices)
    # set_trace()
def train_tokenizer(docs, ent_ids_path, outpath):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    #read list of codes
    ent_ids = []
    try:
        with open(ent_ids_path, "r") as fi:
            ent_ids = [x.replace("\n","") for x in fi.readlines()]    
    except FileNotFoundError:
        print("Could not find file with clinical entities")
    special_tokens = ["[PAD]", "[MASK]", "[UNK]", "[CLS]", "[SEP]"] + ent_ids    
    trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=10000)
    
    tokenizer.train_from_iterator(docs, trainer=trainer)
    tokenizer.save(f"{outpath}/tokenizer.json")
    return tokenizer

def cmdline_args():
    parser = argparse.ArgumentParser(description="Generate MLM data from MIMIC notes")
    parser.add_argument('-input', type=str, required=True, help='path to data')    
    parser.add_argument('-dataset', type=str, required=True, help='name of dataset')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-tok_path', type=str,  help='path to a trained tokenizer')        
    parser.add_argument('-max_preds', type=int,  default=MAX_PREDS, 
                        help='max number of predictions per sequence')        
    parser.add_argument('-max_seq_len', type=int, default=MAX_SEQ_LEN, 
                        help='max length of a sequence')                  
    parser.add_argument('-build_tokenizer', action="store_true",
                         help='train tokenizer')       
    parser.add_argument('-create_mlm', action="store_true",
                         help='create mlm sequences')       
    parser.add_argument('-vectorize', action="store_true",
                         help='vectorize mlm sequences')  
    parser.add_argument('-anno', action="store_true",
                         help='use annotated data')  
    return parser.parse_args()	
    
if __name__ == "__main__":
    args = cmdline_args()    
    assert args.tok_path is not None or args.build_tokenizer
    tokenizah = None    
    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    if args.build_tokenizer:
        print("> training tokenizer")
        df = pd.read_csv(f"{args.input}{args.dataset}.csv", sep="\t")
        docs = list(df["TEXT"])
        if args.anno:
            df_anno = pd.read_csv(f"{args.input}{args.dataset}_anno.csv", sep="\t")    
            docs += list(df_anno["ANNO_TEXT"])

        tokenizah = train_tokenizer(docs, args.input+"/ent_ids.txt", args.output)
    
    if args.create_mlm:
        print("> create MLM data")        
        if not tokenizah:
            tokenizah = Tokenizer.from_file(args.tok_path)
        if args.anno:
            fpath = f"{args.input}{args.dataset}_anno.csv"
        else:
            fpath = f"{args.input}{args.dataset}.csv"            
        create_mlm_data(fpath, args.output, args.dataset,tokenizah, max_seq_len=args.max_seq_len, max_preds=args.max_preds,anno=args.anno)
        
    if args.vectorize:
        print("> vectorize MLM data")
        if not tokenizah:
            tokenizah = Tokenizer.from_file(args.tok_path)
        vectorize(args.output, args.dataset, tokenizah, max_seq_len=args.max_seq_len, 
                    max_preds=args.max_preds)

# notes_path = "/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/DATA/input/mini_full_notes_ner.csv"    
# features_path = "/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/DATA/pretrain/"

# tok_path = "/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/DATA/input/tokenizer.json"
# tokenizah = Tokenizer.from_file(tok_path)



