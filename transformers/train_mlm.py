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

def main(conf, vocab_size, device, train_data, test_data, val_data):    
    
    print(f"vocab size: {vocab_size}")
    model = TransformerModel(vocab_size, conf=conf, device=device).to(device)

    best_val_loss = float("inf")
    epochs = conf["epochs"] # The number of epochs
    best_model = None
    model = model.to(device)
    print("> training")
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
    
    test_loss = best_model.evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    return best_model

def cmdline_args():
    parser = argparse.ArgumentParser(description="train MLM")
    parser.add_argument('-input', type=str, required=True, help='path to data')    
    parser.add_argument('-output', type=str, required=True, help='path to output')    
    parser.add_argument('-tok_path', type=str, required=True, 
                        help='path to a trained tokenizer')        
    parser.add_argument('-conf_path', type=str, required=True, 
                        help='path to a config file')        
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

    tokenizah = Tokenizer.from_file(args.tok_path)

    with open(args.conf_path) as fi:
        conf = json.load(fi)
    pin_mem = conf["pin_memory"] if conf["pin_memory"] else False
    n_workers = conf["data_loader_workers"] if conf["data_loader_workers"] else 4
    bsize = conf["batch_size"]
    train_data = DataLoader(MLMDataset(args.input, "train"), batch_size=bsize,
                            shuffle=True, num_workers=4, pin_memory=pin_mem)
    test_data = DataLoader(MLMDataset(args.input, "test"), shuffle=False, num_workers=4)
    val_data = DataLoader(MLMDataset(args.input, "val"), shuffle=False, num_workers=4)
    # the size of vocabulary
    ntokens = len(tokenizah.get_vocab()) 
    main(conf, ntokens, device, train_data, test_data, val_data)

