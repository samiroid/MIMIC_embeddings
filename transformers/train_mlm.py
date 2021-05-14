import math
import torch
import torch.nn as nn
import torch

import argparse
import json
import time
from ipdb import set_trace
import pandas as pd
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
import os
from mytransformer import TransformerModel, MLMDataset, save_model, load_model

def cmdline_args():
    parser = argparse.ArgumentParser(description="train MLM")
    parser.add_argument('-input', type=str, required=True, help='path to data')    
    parser.add_argument('-dataset', type=str, required=True, help='name of the dataset')    
    parser.add_argument('-output', type=str, required=True, help='path to output')    
    parser.add_argument('-tok_path', type=str, required=True, 
                        help='path to a trained tokenizer')        
    parser.add_argument('-load', type=str, help='path to trained model')        
    parser.add_argument('-conf_path', type=str, required=True, 
                        help='path to a config file')        
    parser.add_argument('-train', action="store_true",
                         help='train model')      
    parser.add_argument('-test', action="store_true",
                         help='test model')      
    parser.add_argument('-device', type=str, default="auto", help='device')
    return parser.parse_args()	

def main():

    args = cmdline_args()
    dirname = os.path.dirname(args.output)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    device = None
    if args.device == "auto":
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = None
    
    if args.train:
        print("> training")
        with open(args.conf_path) as fi:
            conf = json.load(fi)
        #fix random seed
        torch.manual_seed(conf["random_seed"])
        run_id = conf["id"]
        pin_mem = conf["pin_memory"] if conf["pin_memory"] else False
        n_workers = conf["data_loader_workers"] if conf["data_loader_workers"] else 4
        bsize = conf["batch_size"]
        data_path = args.input + args.dataset
        train_data = DataLoader(MLMDataset(data_path, "train"), batch_size=bsize,
                                shuffle=True, num_workers=n_workers, pin_memory=pin_mem)    
        val_data = DataLoader(MLMDataset(data_path, "val"), shuffle=False,
                                        num_workers=n_workers)
        tokenizah = Tokenizer.from_file(args.tok_path)
        # the size of vocabulary
        vocab_size = len(tokenizah.get_vocab()) 
        print(f"vocab size: {vocab_size}")
        model = TransformerModel(vocab_size, conf=conf, device=device).to(device)    
        model = model.to(device)    
        # train model
        model.fit(train_data, val_data)    
        outpath = f"{args.output}model_{args.dataset}_{run_id}.pt"
        save_model(model, outpath)
    
    if args.test:
        print("> test")
        if not model:                        
            assert args.load 
            print("loading model @ {args.load}")
            model = load_model(args.load, device)
            model = model.to(device)
        data_path = args.input + args.dataset
        test_data = DataLoader(MLMDataset(data_path, "test"), shuffle=False, num_workers=4)
        test_loss = model.evaluate(test_data)
        print('=' * 89)
        print('test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)    

if __name__ == "__main__":
    main()
