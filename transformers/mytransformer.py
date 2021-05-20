import math
from ipdb import set_trace
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import pprint
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler 
import collections
import itertools

def colstr(st, color, best=False):    
    if best:
        cstring = "\033[36m" + st  + "\033[0m"
    elif color == 'red':
        cstring = "\033[31m" + st  + "\033[0m"
    elif color == 'green':    
        cstring = "\033[32m" + st  + "\033[0m"
    else:
        cstring = "\033[37m" + st  + "\033[0m"    
    return cstring   

class MLMDataset(Dataset):  
  def __init__(self, fname, partition):
        assert partition in ["train","test","val"]
        self.tokens = np.load(f"{fname}_{partition}_tokens.npy")
        self.labels = np.load(f"{fname}_{partition}_labels.npy")
        self.attention_masks = np.load(f"{fname}_{partition}_mask.npy")
        self.x = np.load(f"{fname}_{partition}_x.npy")
        
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

    def __init__(self, d_model, dropout=1.0, max_len=768):
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
        "id": "test",
        "random_seed": 143,
        "emb_size": 200,
        "hidden_size": 200,
        "nhead": 8,
        "nlayers": 4,
        "dropout": 0.2,
        "lr": 5.0,
        "step_size": 1.0,
        "clipgrad_norm":0.5,
        "decay_gamma": 0.95,
        "pin_memory": False,
        "batch_size": 16,
        "data_loader_workers": 4,
        "epochs": 6,
        "warmup_steps":10,
}

    def __init__(self, vocab_size, conf=DEFAULT_CONF, device=None, 
                checkpoint_path=None, train_log_path=None, checkpoint_step=0):
        super(TransformerModel, self).__init__()        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #create logs and checkpoints directories
        self.checkpoint_path = checkpoint_path
        self.checkpoint_step = checkpoint_step
        if self.checkpoint_step > 0:             
            print(f"saving checkpoints @ {self.checkpoint_path}")            
            dirname = os.path.dirname(checkpoint_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
        self.train_log_path = train_log_path        
        if train_log_path:
            dirname = os.path.dirname(train_log_path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            print(f"saving train logs @ {self.train_log_path}")
            
            self.tensorboard = SummaryWriter(f"{self.train_log_path}")
        else:
            print("no train log")
            self.tensorboard = None
            

        print(f"> init transformer @ {self.device}")
        pprint.pprint(conf)                
        self.first_epoch = 1
        self.last_batch_idx = -1
        self.conf = conf
        self.clipgrad_norm = conf["clipgrad_norm"] 
        self.vocab_size=vocab_size
        self.epochs = conf["epochs"]
        self.lr = conf["lr"]
        self.emb_size = conf["emb_size"]
        self.pos_encoder = PositionalEncoding(self.emb_size, conf["dropout"])        
        encoder_layers = TransformerEncoderLayer(self.emb_size, conf["nhead"], 
                                                conf["hidden_size"], conf["dropout"])
        self.transformer_encoder = TransformerEncoder(encoder_layers, conf["nlayers"])
        self.embedding = nn.Embedding(self.vocab_size, self.emb_size)
        self.decoder = nn.Linear(self.emb_size, self.vocab_size)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        #replicating optimizer from original BERT
        #https://github.com/google-research/bert/blob/master/optimization.py
        weight_decay_rate=0.01
        beta_1=0.9
        beta_2=0.999
        epsilon=1e-6
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, eps=epsilon,
                                            betas=(beta_1, beta_2),
                                            weight_decay=weight_decay_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, conf["step_size"], 
                                                        conf["decay_gamma"])
        self.scheduler_warmup = GradualWarmupScheduler(self.optimizer, 
                                                        total_epoch=conf["warmup_steps"],               
                                                        multiplier=1,
                                                        after_scheduler=self.scheduler)

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

    def fit(self, train_dataloader, val_dataloader):
        self.train() # Turn on the train mode
        prev_train_loss = float("inf")
        prev_val_loss = float("inf")        
        best_val_loss = float("inf")        
        best_train_loss = float("inf")        
        train_loss = 0.
        val_loss = 0.
        cur_loss = 0.
        val_loss_str = "0.00"
        best_model = None            
        
        #if a checkpoint has been loaded fastforward dataloader to last batch
        try:
            if self.last_batch_idx:                
                print(f"fast forward to batch: {self.last_batch_idx}")
                # set_trace()                
                # for i, _ in enumerate(train_dataloader):
                #     if i >= self.last_batch_idx:
                #         break
                    # print(len(z))
        except AttributeError:
            pass
            
        print("buga")
        #number of total batches seen by the model
        global_step = 0                
        #self.first_epoch has the epoch we start from (in case a checkpoint was loaded)        
        for epoch in range(self.first_epoch, self.epochs + 1):
            #warmup scheduler needs to take step before optimizer
            #this will raise a UserWarning
            self.scheduler_warmup.step()
            with tqdm(train_dataloader, unit="bt") as tepoch:
                tepoch.set_description(f"Epoch: {epoch}")                
                curr_lr = self.optimizer.param_groups[0]['lr']
                if curr_lr < 0.1:
                    curr_lr = "{:.2E}".format(self.scheduler.get_last_lr()[0])
                for i, batch in enumerate(tepoch):
                    if i <= self.last_batch_idx and self.last_batch_idx > -1:
                        # print(f"skip {i}")
                        tepoch.set_description(f"fast forward {i}")
                        continue
                    #load batch and send to device
                    seq, X, Y, src_mask = batch            
                    seq = seq.to(self.device)
                    Y = Y.to(self.device)
                    src_mask = src_mask.to(self.device)                    
                    X = X.to(self.device)
                    self.optimizer.zero_grad()            
                    output = self.forward(seq)            
                    #0,1,2,...,bsize
                    batch_idx = torch.arange(output.shape[0]).long().unsqueeze(1).to(self.device)                    
                    #select only the indices corresponding to the MLM predictions
                    try:
                        preds = output[batch_idx,X,:].view(-1, self.vocab_size)      
                    except IndexError as e:
                        print(e)
                        set_trace()
                    loss = self.criterion(preds, Y.view(-1))
                    try:
                        loss.backward()
                    except RuntimeError:
                        set_trace()

                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipgrad_norm)
                    try:
                        self.optimizer.step()               
                    except RuntimeError as e:
                        print(e)
                        set_trace()
                    train_loss += loss.item()                
                    
                    global_step+=1
                    cur_loss = train_loss / ((i+1)*train_dataloader.batch_size)
                    cur_loss = round(cur_loss, 3)
                    if self.tensorboard:
                        self.tensorboard.add_scalar("batch/loss", cur_loss, global_step)                        
                    
                    if cur_loss > prev_train_loss:
                        train_loss_str = colstr(str(cur_loss),"red")
                    elif cur_loss < prev_train_loss:                        
                        train_loss_str = colstr(str(cur_loss),"green")
                    else:
                        train_loss_str = colstr(str(cur_loss),"")
                    tepoch.set_postfix(tr_loss=train_loss_str, 
                                        val_loss=val_loss_str,
                                        lr=curr_lr)               
                    if i > 0 and self.checkpoint_step > 0 and i % self.checkpoint_step == 0 :
                        params = best_model if best_model else self.state_dict()
                        self.save_checkpoint(epoch, best_val_loss, params, i)                        

                prev_train_loss = cur_loss
                train_loss = 0.
                val_loss = self.evaluate(val_dataloader)                  
                val_loss = round(val_loss,3)

                if val_loss > prev_val_loss:
                    val_loss_str = colstr(str(val_loss),"red")
                elif val_loss < prev_val_loss:
                    best=False
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = self.state_dict()
                        best = True
                    val_loss_str = colstr(str(val_loss),"green",best)
                else:
                    val_loss_str = str(val_loss)   
                prev_val_loss = val_loss                
                
                tepoch.set_postfix(tr_loss=train_loss_str, val_loss=val_loss_str, lr=curr_lr)
                if self.tensorboard:                    
                    self.tensorboard.add_scalar("train/loss", cur_loss, epoch)
                    self.tensorboard.add_scalar("val/loss", val_loss, epoch)
                    for name, weight in self.named_parameters():
                        self.tensorboard.add_histogram("weights/"+name,weight, epoch)
                        self.tensorboard.add_histogram("grads/"f'{name}.grad',weight.grad, epoch)
                #save a checkpoint
                self.save_checkpoint(epoch, best_val_loss, best_model, 0)
        #load best parameters
        self.load_state_dict(best_model)


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
                #0,1,2,...,bsize
                batch_idx = torch.arange(output.shape[0]).long().unsqueeze(1)
                #select only the indices corresponding to the MLM predictions
                preds = output[batch_idx,X,:].view(-1, self.vocab_size)                
                total_loss += len(data) * self.criterion(preds, Y.view(-1)).item()
        return total_loss / (len(dataloader.dataset) - 1)

    def save_checkpoint(self, epoch, loss, best_model,last_batch_idx):
        if self.checkpoint_path:            
            chkpt = {
                'epoch': epoch,
                'loss': loss,            
                'model_state_dict': best_model,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'last_batch_idx':last_batch_idx
            }
            torch.save(chkpt, self.checkpoint_path)      
        else:
            print("=> could not save checkpoint '{}'".format(self.checkpoint_path))

    def load_checkpoint(self):          
        if self.checkpoint_path and os.path.isfile(self.checkpoint_path):            
            checkpoint = torch.load(self.checkpoint_path)    
            self.to(self.device)        
            self.first_epoch = checkpoint['epoch']
            self.last_batch_idx = checkpoint["last_batch_idx"]
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})" .format(self.checkpoint_path, self.first_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(self.checkpoint_path))
        

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
    