import torch
import os
import argparse


# allows us to take in True/False values without defaulting to True on String
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# takes the input arguments, Strings in add_argument() are what to type into CLI
def get_ph_inputs_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", type=str)
    parser.add_argument("--split", type=str)
    parser.add_argument("--sub", type=str2bool, default=False)

    args = parser.parse_args()
    score_var, split, sub = args.score, args.split, args.sub

    print('ARGS ARE:')
    print(f"score_var: {score_var}, split: {split}, sub: {sub}")

    return score_var, split, sub


def init_bert_for_embeddings():
    from transformers import BertModel, BertTokenizer
    model_name = 'bert-base-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name, max_length=512, padding="max_length", truncation=True)
    # load
    model = BertModel.from_pretrained(model_name)
    return model, tokenizer

def get_bert_embedding(s, model, tokenizer):
    input_ids = tokenizer.encode(s, add_special_tokens=True, max_length=512, padding="max_length", truncation=True)
    input_ids = torch.tensor([input_ids])

    with torch.no_grad():
        last_hidden_states = model(input_ids)[0] # Models outputs are now tuples
    # creates sentence embedding of 768 by averaging across tokens
    last_hidden_states = last_hidden_states.mean(1)

    return last_hidden_states



# start at base -- run_pyhard.py
# DATA LOADING
THIS_SVAR, THIS_SPLIT, subset, = get_ph_inputs_from_args()
# THIS_SVAR = 'TrustPhys'
# THIS_SPLIT = 'train'
# subset=False

# import from parent folder
from utils.data_loading import *

if THIS_SVAR == 'wer':
    print('using ZDA data')
    input_data_d = load_zda_data('data/data_zda/', subset=subset)
else:
    print('using HAL data')
    input_data_d = load_hal_data('data/DataCVFolds/', score_variable=THIS_SVAR, subset=subset)

if 'train' in THIS_SPLIT:
    from utils.data_processing import *
    input_data_d['hard_train_data'], input_data_d['rand_train_data'] = sample_hard_rand(input_data_d['train_data'], THIS_SVAR)


# now we have everything so navigate to subdir
sub_dir = '/'.join(['pyhard', THIS_SVAR, THIS_SPLIT])
os.chdir( sub_dir )


# BERT EMBEDDINGS
THIS_DF = input_data_d[THIS_SPLIT+'_data']
THIS_TEXT = [ t for t in  THIS_DF['text']]
model, tokenizer = init_bert_for_embeddings()

# create the input
EMBS = np.array([get_bert_embedding(s, model, tokenizer).squeeze().numpy() for s in THIS_TEXT])
THIS_TARGET = pd.Series( [ int(v==1) for v in THIS_DF['label'] ], name='y')
# target col needs to be last for default config
input_df = pd.concat([ pd.DataFrame(EMBS), THIS_TARGET ], axis=1)

# data.csv is name for default config
input_df.to_csv('data.csv', index=False)

# run the pyhard commands -- should do the rest
# !pyhard init
# !pyhard run
os.system("pyhard init")
os.system("pyhard run")
