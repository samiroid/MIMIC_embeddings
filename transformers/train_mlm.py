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
from mytransformer import TransformerModel, MLMDataset

def train(conf, vocab_size, device, train_data, val_data):            
    model = TransformerModel(vocab_size, conf=conf, device=device).to(device)
    best_val_loss = float("inf")
    epochs = conf["epochs"] # The number of epochs
    best_model = None
    model = model.to(device)    
    # loop over epochs
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        model.fit(train_data)
        val_loss = model.evaluate(val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        val_loss, math.exp(val_loss)))
        print('-' * 89)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model        
    
    test(best_model, test_data)
    # test_loss = best_model.evaluate(test_data)
    # print('=' * 89)
    # print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    #     test_loss, math.exp(test_loss)))
    # print('=' * 89)

    return best_model

def test(model, test_data):
    
    test_loss = model.evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    return test_loss

def cmdline_args():
    parser = argparse.ArgumentParser(description="train MLM")
    parser.add_argument('-input', type=str, required=True, help='path to data')    
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

if __name__ == "__main__":
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
        pin_mem = conf["pin_memory"] if conf["pin_memory"] else False
        n_workers = conf["data_loader_workers"] if conf["data_loader_workers"] else 4
        bsize = conf["batch_size"]
        train_data = DataLoader(MLMDataset(args.input, "train"), batch_size=bsize,
                                shuffle=True, num_workers=n_workers, pin_memory=pin_mem)    
        val_data = DataLoader(MLMDataset(args.input, "val"), shuffle=False,
                                        num_workers=n_workers)
        tokenizah = Tokenizer.from_file(args.tok_path)
        # the size of vocabulary
        vocab_size = len(tokenizah.get_vocab()) 
        print(f"vocab size: {vocab_size}")
        model = train(conf, vocab_size, device, train_data, val_data)
    
    if args.test:
        print("> test")
        if not model:
            model = None #load model
            print("need model...")
        else:
            test_data = DataLoader(MLMDataset(args.input, "test"), shuffle=False, num_workers=4)

