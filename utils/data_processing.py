import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import roc_auc_score


###################################################################
################### Dataset for Transformers ######################
###################################################################
# used for efficient testing on subsets
def stratified_sample(df, n):
    return pd.concat( [ df[df.label!=1].sample(n=round(n/2), random_state=123),
                df[df.label==1].sample(n=round(n/2), random_state=123) ] ).sample(
                    frac=1, random_state=123).reset_index(drop=True)

# Create a custom Dataset class for your data
class TextClassificationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length):
        dataframe.label = [ 1 if v==1 else 0 for v in dataframe.label ]
        self.data = dataframe
        self.texts = dataframe.text
        self.labels = dataframe.label
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ID = dataframe.ID

        self.score = pd.Series([0]*len(dataframe), name='score')
        self.EMB_ID = 'B'

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'ID': self.ID[idx],
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
        }



def prepare_data(input_data_d, tokenizer, emb_type, max_length, dataset_param_d=None, input_data_struct_d=None):

    # check or train w2v on train
    if emb_type == 'HF':
        hard_train_dataset = TextClassificationDataset(input_data_d['hard_train_data'], tokenizer, max_length)
        rand_train_dataset = TextClassificationDataset(input_data_d['rand_train_data'], tokenizer, max_length)
        # do embeddings and labels for val and test
        val_dataset = TextClassificationDataset(input_data_d['val_data'], tokenizer, max_length)
        test_dataset = TextClassificationDataset(input_data_d['test_data'], tokenizer, max_length)
    else:
        hard_train_dataset = Word2VecDataset(input_data_d['hard_train_data'], tokenizer, set_type='train', MAX_LEN=256, EMBEDDING_DIMENSION=32)
        rand_train_dataset = Word2VecDataset(input_data_d['rand_train_data'], tokenizer, set_type='train', MAX_LEN=256, EMBEDDING_DIMENSION=32)

        val_dataset = Word2VecDataset(input_data_d['val_data'], tokenizer, set_type='val', MAX_LEN=256, EMBEDDING_DIMENSION=32)
        test_dataset = Word2VecDataset(input_data_d['test_data'], tokenizer, set_type='test', MAX_LEN=256, EMBEDDING_DIMENSION=32)
    return {
        'hard_train_set': hard_train_dataset,
        'rand_train_set': rand_train_dataset,
        'val_set': val_dataset,
        'test_set': test_dataset
    }


def sample_hard_rand(tdf, score_variable):
    this_split_path = f'splits/{score_variable}_train_ids.csv'
    if os.path.isfile(this_split_path)==False:
        # print('MAKING NEW TRAIN SPLIT')
        # first time running, need to make splits from scratch
        case_df = tdf[tdf['label']==1]
        N_case = len(case_df)
        ctrl_df = tdf[tdf['label']!=1]
        N_ctrl = len(case_df)

        hard_ctrl = ctrl_df.sort_values('score', ascending=False).head(round(N_case/2))
        hard_case = case_df.sort_values('score', ascending=True).head(round(N_ctrl/2))
        hard_df = pd.concat([ hard_ctrl, hard_case ]).reset_index(drop=True)

        rand_ctrl = ctrl_df.sample(frac=0.5, random_state=123)
        rand_case = case_df.sample(frac=0.5, random_state=123)
        rand_df = pd.concat([ rand_ctrl, rand_case ]).reset_index(drop=True)

        output_ids = pd.concat([ pd.Series(hard_df['ID'], name='hard_ids'),
                                pd.Series(rand_df['ID'], name='rand_ids') ], axis=1)
        output_ids.to_csv(this_split_path)

    else:
        # print('USING EXISTING TRAIN SPLITS')
        input_ids = pd.read_csv(this_split_path)
        hard_df = tdf[tdf['ID'].isin(input_ids['hard_ids'])]
        rand_df = tdf[tdf['ID'].isin(input_ids['rand_ids'])]

    return hard_df.reset_index(drop=True), rand_df.reset_index(drop=True)


def use_hard_data(DIFFICULT_TIME, epoch):
    if DIFFICULT_TIME != 'None':
        if DIFFICULT_TIME == 'Constant':
            return True
        elif DIFFICULT_TIME == 'First' and epoch <= 5:
            return True
        elif DIFFICULT_TIME == 'Second' and epoch > 5:
            return True
        else:
            return False
    else:
        return False

def set_data_loaders(these_params, epoch, hard_train_dataset, rand_train_dataset,
                     val_dataset, test_dataset, ):
    # pick which training set initially
    if use_hard_data(these_params['difficultTime'], epoch):
        print('using hard data...')
        train_dataloader = DataLoader(hard_train_dataset, batch_size=these_params['batchSize'], shuffle=True)
    else:
        print('using random data...')
        train_dataloader = DataLoader(rand_train_dataset, batch_size=these_params['batchSize'], shuffle=True)

    val_dataloader = DataLoader(val_dataset, batch_size=these_params['batchSize'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=these_params['batchSize'], shuffle=False)

    return {
        'train_loader': train_dataloader,
        'val_loader': val_dataloader,
        'test_loader': test_dataloader
    }




import gensim
from nltk.tokenize import word_tokenize


def make_embeddings(vocab_d, token_lst, vec_len):
    return [ np.array([ vocab_d[w] if w in vocab_d.keys() else [0]*vec_len for w in s]) for s in token_lst ]

def pt_sentence(this_sent, max_len, emb_len):
    if len(this_sent) < max_len:
        diff = max_len - len(this_sent)
        pad = np.zeros((diff, emb_len))
        if len(this_sent) > 0:
            new_sent = np.append(this_sent, pad, axis=0)
        else:
            new_sent = pad
    elif len(this_sent) > max_len:
        new_sent = this_sent[:max_len, :]

    return new_sent

# pad/truncate numpy array of embeddings to max length
def pad_truncate(emb_lst, max_len, emb_len):
    print('padding and truncating...')

    pt_embeddings = []
    for this_sent in emb_lst:
        if len(this_sent) < max_len:
            diff = max_len - len(this_sent)
            pad = np.zeros((diff, emb_len))
            if len(this_sent) > 0:
                new_sent = np.append(this_sent, pad, axis=0)
            else:
                new_sent = pad
        elif len(this_sent) > max_len:
            new_sent = this_sent[:max_len, :]
        pt_embeddings.append(new_sent)

    return np.array(pt_embeddings)




###################################################################
################### Dataset for Local NNs #########################
###################################################################
# simple helper used in checking for existing w2v embs
def check_file_exists(path):
    return os.path.exists(path)

# w2v dataset used for all non-transformer models
class Word2VecDataset(Dataset):
    def __init__(self, dataframe, tokenizer, set_type='train', struct_df=None,
                MAX_LEN=512, EMBEDDING_DIMENSION=300,
                WINDOW_SIZE=3, MIN_FREQUENCY_THRESHOLD=2, IS_SKIPGRAM=1,
                IS_LOWER_WORDS=True, ITERATIONS=20, strat=None):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.data.label = np.array([ l if l==1 else 0 for l in self.data.label ])
        self.struct_data = struct_df
        self.set_type = set_type
        # this is the column of text
        self.texts = dataframe.text
        # this is the class
        # NOTE; changing this to store test losses during training
        # self.labels = self.data.label if set_type != 'test' else []
        self.labels = self.data.label
        self.score = dataframe.score
        self.ID = dataframe.ID

        self.max_len = MAX_LEN
        self.vec_len = EMBEDDING_DIMENSION
        self.win_size = WINDOW_SIZE
        self.min_freq = MIN_FREQUENCY_THRESHOLD
        self.is_skg = IS_SKIPGRAM
        self.is_lower = IS_LOWER_WORDS
        self.iter = ITERATIONS

        # set param names and tokenize
        self.emb_name = f"w2v-{MAX_LEN}_{EMBEDDING_DIMENSION}_{WINDOW_SIZE}_{MIN_FREQUENCY_THRESHOLD}_{IS_SKIPGRAM}_{IS_LOWER_WORDS}_{ITERATIONS}"
        self.sentences = [ [ t.lower() for t in self.tokenizer(s) ] if self.is_lower else self.tokenizer(s) for s in self.texts]
        print(f"{self.set_type} -- {self.emb_name}")

        # if we don't have these embeddings, create them!
        if check_file_exists(f'data/embeddings/{self.emb_name}.txt')==False:
            # BUT only for train, else error
            if self.set_type == 'train':
                print('training new w2v model...')
                self.train_w2v_model()
                self.write_w2v_embs_file()
            else:
                raise ValueError("set_type != 'train' and missing word2vec model")

        # ALL train, val, and test need to run embedding step
        self.vocab_d = self.read_w2v_embs_file()
        self.EMB_ID = self.get_emb_id()
        embeddings = make_embeddings(self.vocab_d, self.sentences, self.vec_len)
        self.pt_embeddings = pad_truncate(embeddings, self.max_len, self.vec_len)

    # need to train a model if we don't have the embs saved locally
    def train_w2v_model(self):
        # need to train a w2v model if not using pretrained
        self.w2v_model = gensim.models.Word2Vec(
            self.sentences,
            sg=self.is_skg,
            vector_size=self.vec_len,
            min_count=self.min_freq,
            window=self.win_size,
            epochs=self.iter,
            compute_loss=True,
            seed=123
        )

    # write the created embs to a cached file for re-runs
    def write_w2v_embs_file(self):
        try:
            # add an ID to the ledger
            with open(f'data/embeddings/EMB_LEDGER.txt', 'r') as f:
                num_lines = len(f.readlines())
        except FileNotFoundError:
            with open(f'data/embeddings/EMB_LEDGER.txt', 'w') as f:
                pass
            num_lines = 0
        
        # print(f"writing {self.emb_name} {num_lines+1}")
        with open(f'data/embeddings/EMB_LEDGER.txt', 'a') as f:
            f.write(f"{self.emb_name} {num_lines+1}\n")

        all_words = list(set([item for sublist in self.sentences for item in sublist]))
        with open(f'data/embeddings/{self.emb_name}.txt', 'a') as f:
            for word in all_words:
                this_emb = self.w2v_model.wv[word] if word in self.w2v_model.wv.key_to_index.keys() else np.array([0]*self.vec_len)
                f.write(f"{word} ")
                for v in this_emb:
                    f.write(f"{v} ")
                f.write('\n')

    # we have a ledger of embeddings with IDs so we don't have to include all params in model_name
    def get_emb_id(self):
        # read in the ledger
        with open(f'data/embeddings/EMB_LEDGER.txt', 'r') as f:
            lines = f.read().splitlines()

        # convert to dict
        this_d = {}
        for line in lines:
            this_split = line.split(' ')
            this_d[this_split[0]] = str(this_split[1])
        
        # find the right item
        return this_d[self.emb_name]

    # read in an existing w2v embs file
    def read_w2v_embs_file(self):
        print('reading in embeddings...')
        with open(f'data/embeddings/{self.emb_name}.txt', 'r') as f:
            lines = f.read().splitlines()
            
        this_d = {}
        for line in lines:
            this_split = line.split(' ')
            this_d[this_split[0]] = np.array(this_split[1:-1]).astype(float)

        return this_d

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return {
            'embeddings': self.pt_embeddings[index],
            'labels': self.labels[index],
            'ID': self.data.ID[index],
        }


###################################################################
################### Inference and Metrics #########################
###################################################################
# custom metrics function for keras / sklearn
def calculate_metrics(model, thisData, theseLabels, is_val=False):
    # this matches TEST where we take CASE [1] instead of CONTROL [0]
    these_probs = [e[1] for e in model.predict(thisData)]
    this_auc = roc_auc_score(theseLabels, these_probs)
    these_preds = [int(e > 0.5) for e in these_probs]

    # .fit() for train already returns loss, acc so only calculate for inference
    if is_val:
        this_loss, this_acc = model.evaluate(thisData, theseLabels)
        return (this_loss, this_acc, this_auc, these_probs, these_preds, theseLabels)
    # avoid repeated compute for train
    else:
        return these_probs, this_auc, these_preds


# used for local NNs to generate probs
def run_inference(model, thisData):
    these_probs = [e[1] for e in model.predict(thisData)]
    return these_probs


# breaks out of training loop if no improvement in train loss for 4 epochs
def check_patience(train_losses):
    if len(train_losses) >= 4:
        if ((train_losses[-1] >= train_losses[-2]) and
            (train_losses[-2] >= train_losses[-3]) and
            train_losses[-3] >= train_losses[-4]):
            print("PATIENCE REACHED")
            return True
    return False