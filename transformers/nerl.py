import argparse
import pandas as pd 
import medcat
from medcat.cat import CAT
from medcat.utils.vocab import Vocab
from medcat.cdb import CDB 
import os

from tokenizers import Tokenizer, decoders
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from ipdb import set_trace


cat = None
ent_ids = []
def load_cat(vocab_path, cdb_path ):
    #https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt
    # Sign or Symptom | T184
    # Laboratory or Test Result | T034
    # Health Care Activity | T058
    # Diagnostic Procedure | T060
    # Antibiotic | T195
    # Pharmacologic Substance | T121
    # Disease or Syndrome | T047
    # Pathologic Function | T046

    vocab = Vocab()
    vocab.load_dict(vocab_path)
    print("Loaded Vocab")
    cdb = CDB()
    cdb.load_dict(cdb_path) 
    print("Loaded CDB")
    # create cat
    global cat
    cat = CAT(cdb=cdb, vocab=vocab)
    #filter entity types
    ENTITY_TYPES = ['T047', 'T048', 'T184','T034','T058','T060','T195','T121','T047','T046']
    cat.spacy_cat.TUI_FILTER = ENTITY_TYPES    
    print("Loaded CAT")

def annotate(txt):    
    global ent_ids
    entities = cat.get_entities(txt)    
    new_text = txt[:]
    for ent in entities:
        span = ent["source_value"]
        cui = "["+ent["cui"]+"]"
        tui = "["+ent["tui"]+"]"
        anno = f"{tui} {cui}"
        ent_ids.append(tui)
        ent_ids.append(cui)
        new_text = new_text.replace(f" {span} ",f" {anno} " )
    return new_text

def train_tokenizer(docs, ent_ids_path, outpath):
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.decoder = decoders.WordPiece()
    #read list of codes
    with open(ent_ids_path, "r") as fi:
        ent_ids = [x.replace("\n","") for x in fi.readlines()]
    
    special_tokens = ["[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"] + ent_ids
    print(special_tokens)
    trainer = WordPieceTrainer(special_tokens=special_tokens, vocab_size=5000)
    
    tokenizer.train_from_iterator(docs, trainer=trainer)
    tokenizer.save(f"{outpath}/tokenizer.json")
    from pdb import set_trace; set_trace()

def cmdline_args():
    parser = argparse.ArgumentParser(description="Clinical NER+L over MIMIC notes")
    parser.add_argument('-input', type=str, required=True, help='path to notes')    
    parser.add_argument('-output', type=str, required=True, help='path of the output')    
    parser.add_argument('-medcat_vocab', type=str, required=True, help='path to medcat Vocab file')       
    parser.add_argument('-medcat_cdb', type=str, required=True, help='path to medcat CDB file')       
    parser.add_argument('-ner', action="store_true", help='do ner')       
    parser.add_argument('-build_tokenizer', action="store_true", help='do ner')       
    return parser.parse_args()	
    
if __name__ == "__main__":
    args = cmdline_args()
    outdir = os.path.dirname(args.output)
    if args.ner:
        load_cat(args.medcat_vocab, args.medcat_cdb )
        df = pd.read_csv(args.input, sep="\t")
        df["ANNO_TEXT"] = df["TEXT"].apply(annotate)
        df[["SUBJECT_ID","ANNO_TEXT"]].to_csv(args.output, index=False, sep="\t", header=True)    
        print(ent_ids)
        with open(outdir+"/ent_ids.txt", "w") as fo:        
            fo.write("\n".join(list(set(ent_ids))))
    
    if args.build_tokenizer:
        print("training tokenizer")
        df = pd.read_csv(args.input, sep="\t")
        df_anno = pd.read_csv(args.output, sep="\t")    
        docs = list(df_anno["ANNO_TEXT"]) + list(df["TEXT"])
        train_tokenizer(docs, outdir+"/ent_ids.txt", outdir)
