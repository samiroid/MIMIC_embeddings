import argparse
from ipdb import set_trace
import numpy as np
from numpy import arange
import pandas as pd
import math
from numpy.random import RandomState
import os
from tqdm import tqdm, trange
rng = RandomState(123)

def cmdline_args():
    parser = argparse.ArgumentParser(description="split mimic notes into train and val")
    parser.add_argument('-input', type=str, required=True, help='path to notes')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-split', type=float, default=0.2, help='split')    

    return parser.parse_args()

if __name__ == "__main__":
    args = cmdline_args()
    fname = os.path.split(args.input)[1]
    df = pd.read_csv(args.input, sep="\t")
    
    df_train = pd.DataFrame(columns=df.columns.tolist())
    df_val = pd.DataFrame(columns=df.columns.tolist())    
    for i in trange(len(df)): 
        row = df.iloc[i]
        notes = row["TEXT"].split(" [EON] ")
        N = len(notes)
        n = math.floor(N*args.split)        
        idxs = rng.permutation(N)
        notes = np.array(notes)
        val_notes = notes[idxs[:n]].tolist()
        train_notes = notes[idxs[n:]].tolist()
        df_train.loc[i] = [row["SUBJECT_ID"], " [EON] ".join(train_notes)]   
        if len(val_notes) > 0:     
            df_val.loc[i] = [row["SUBJECT_ID"], " [EON] ".join(val_notes)]
    df_train.to_csv(f"{args.output}train_{fname}", index=False, header=True, sep="\t")
    df_val.to_csv(f"{args.output}val_{fname}", index=False, header=True, sep="\t")
    # print(len(notes))
    # set_trace()
