import math
import torch
import torch.nn as nn
import torch

import time
from ipdb import set_trace
import pandas as pd
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from mytransformer import TransformerModel, MLMDataset



def main(dev=None, pin_mem=False):
    if not dev:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = dev
    
    path = "/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/DATA/pretrain/"
    tok_path = "/home/silvio/home/projects/MIMIC_embeddings/MIMIC_embeddings/DATA/input/tokenizer.json"
    tokenizah = Tokenizer.from_file(tok_path)
    
    train_data = DataLoader(MLMDataset(path, "train"), batch_size=8,
                            shuffle=True, num_workers=4, pin_memory=pin_mem)
    test_data = DataLoader(MLMDataset(path, "test"), shuffle=False, num_workers=4)
    val_data = DataLoader(MLMDataset(path, "val"), shuffle=False, num_workers=4)


    # the size of vocabulary
    ntokens = len(tokenizah.get_vocab()) 
    
    print(f"vocab size: {ntokens}")
    model = TransformerModel(ntokens, device=device).to(device)

    best_val_loss = float("inf")
    epochs = 3 # The number of epochs
    best_model = None

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

main()