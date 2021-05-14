import math
from ipdb import set_trace
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import pprint

class MLMDataset(Dataset):  
  def __init__(self, path, partition):
        assert partition in ["train","test","val"]
        self.tokens = np.load(f"{path}_{partition}_tokens.npy")
        self.labels = np.load(f"{path}_{partition}_labels.npy")
        self.attention_masks = np.load(f"{path}_{partition}_mask.npy")
        self.x = np.load(f"{path}_{partition}_x.npy")
        
  def __len__(self):  
        return len(self.tokens)

  def __getitem__(self, index):  
        # Select sample
        S = torch.from_numpy(self.tokens[index,:])
        X = torch.from_numpy(self.x[index,:]).long()
        Y = torch.from_numpy(self.labels[index,:])
        M = torch.from_numpy(self.attention_masks[index,:]).bool()        
        return S, X, Y, M

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):

    DEFAULT_CONF = {    
        "emb_size": 200,
        "hidden_size": 200,
        "nhead": 4,
        "nlayers": 2,
        "dropout": 0.2,
        "lr": 5.0,
        "step_size":1.0,
        "decay_gamma":0.95
}

    def __init__(self, vocab_size, conf=DEFAULT_CONF, device=None):
        super(TransformerModel, self).__init__()        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"> init transformer @ {self.device}")
        pprint.pprint(conf)        
        self.conf = conf
        self.vocab_size=vocab_size
        self.lr = conf["lr"]
        self.emb_size = conf["emb_size"]
        self.pos_encoder = PositionalEncoding(self.emb_size, conf["dropout"])        
        encoder_layers = TransformerEncoderLayer(self.emb_size, conf["nhead"], 
                                                conf["hidden_size"], conf["dropout"])
        self.transformer_encoder = TransformerEncoder(encoder_layers, conf["nlayers"])
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.decoder = nn.Linear(self.emb_size, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, conf["step_size"], 
                                                        conf["decay_gamma"])
        self.init_weights()

    def init_weights(self, initrange = 0.1):        
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_key_pad_mask=None):        
        output = self.encode(src, src_key_pad_mask=src_key_pad_mask)
        output = self.decoder(output)
                
        return output
    
    def encode(self, src, src_key_pad_mask=None):
        #embed and rescale
        src = self.embedding(src) * math.sqrt(self.emb_size)
        #add positional encoding    
        src = self.pos_encoder(src)
        #transformer blocks
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_pad_mask)
        return output

    def fit(self, dataloader):
        self.train() # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        n_batches = len(dataloader.dataset) // dataloader.batch_size
        
        for i, batch in enumerate(dataloader):
            data, X, Y, src_mask = batch            
            data = data.to(self.device)
            Y = Y.to(self.device)
            src_mask = src_mask.to(self.device)
            self.optimizer.zero_grad()            
            output = self.forward(data)            
            batch_idx = torch.arange(output.shape[0]).long().unsqueeze(1)
            preds = output[batch_idx,X,:].view(-1, self.vocab_size)                        
            loss = self.criterion(preds, Y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()
            log_interval = 5
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print('|{:5d}/{:5d} batches | '
                    'lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                         i, n_batches, self.scheduler.get_last_lr()[0],
                        elapsed * 1000 / log_interval,
                        cur_loss, math.exp(cur_loss)))
                total_loss = 0
                start_time = time.time()
        self.scheduler.step()

    def evaluate(self, dataloader):
        self.eval() # Turn on the evaluation mode
        total_loss = 0.        
        with torch.no_grad():
            for batch in dataloader:
                data, X, Y, src_mask = batch                
                data = data.to(self.device)
                Y = Y.to(self.device)
                src_mask = src_mask.to(self.device)
                output = self.forward(data)
                batch_idx = torch.arange(output.shape[0]).long().unsqueeze(1)
                preds = output[batch_idx,X,:].view(-1, self.vocab_size)
                # output_flat = output.view(-1, self.vocab_size)
                total_loss += len(data) * self.criterion(preds, Y.view(-1)).item()
        return total_loss / (len(dataloader.dataset) - 1)

def save_model(model, path):
    print(f"> saving model @ {path}")
    torch.save({
        "conf": model.conf,
        "vocab_size": model.vocab_size,
        "model_state_dict": model.state_dict()
    }, path)

def load_model(path, device=None):
    print(f"> loading model @ {path}")
    chkpt = torch.load(path)
    model = TransformerModel(chkpt["vocab_size"], chkpt["conf"], device)
    model.load_state_dict(chkpt["model_state_dict"]) 
    return model
    