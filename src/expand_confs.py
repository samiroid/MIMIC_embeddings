import argparse
from ipdb import set_trace
import numpy as np
from numpy import arange
import pandas as pd
import math
from numpy.random import RandomState
import os
from tqdm import tqdm, trange
import pprint 
rng = RandomState(123)
import itertools
import json

def main(path_in, path_out, max_confs=10):
    
    with open(path_in,"r") as fi:
        master_confs = json.load(fi)
    fixed_attrs = {}
    sweep_attrs = {}
    sweep_keys = []
    sweep_values = []
    for k,v in master_confs.items():
        if isinstance(v,list):
            sweep_attrs[k] = v
            sweep_keys.append(k)
            sweep_values.append(v)
        else:
            fixed_attrs[k] = v
    new_confs = []
    sweeps = list(itertools.product(*sweep_values))
    for conf in sweeps:
        nu_conf = fixed_attrs.copy()
        for i,k in enumerate(sweep_keys):
            # nu_conf = fixed_attrs.copy()
            nu_conf[k] = conf[i] 
        new_confs.append(nu_conf)
    
    rng.shuffle(new_confs)

    for i, nc in enumerate(new_confs[:max_confs]):
        nc["run_id"] = i+1
        with open(f"{path_out}_{i+1}.json", "w") as fo:
            json.dump(nc, fo, ensure_ascii=False)
    
def cmdline_args():
    parser = argparse.ArgumentParser(description="explode master config into individual config files")
    parser.add_argument('-input', type=str, required=True, help='path to master config file')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')        
    parser.add_argument('-max_confs', type=int, default=10, help='maximum number of configurations')        

    return parser.parse_args()

if __name__ == "__main__":
    args = cmdline_args()
    main(args.input, args.output, args.max_confs)
    
