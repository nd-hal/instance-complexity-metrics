def auc_group(df):
    y_score = df['probs']
    y_true = df['labels']
    return roc_auc_score(y_true=y_true, y_score=y_score)

def write_exam_results(split_lst, svar_lst, ledger_hardness):

    for svar in svar_lst:
        sdf = ledger_hardness[ ledger_hardness['score_var']==svar ]
        best_val_epoch = sdf[sdf['split']=='val'].groupby('epoch').apply(auc_group, include_groups=False).idxmax()

        exam_res_d = { 'model_names': None, 'IDs': None, 'Z': None } 

        for full_split in split_lst:
            tdf = sdf[ sdf['split_full']==full_split ]
            these_IDs = pd.Series( tdf['ID'].unique(), name='ID' )
            these_mns = pd.Series( tdf['model_name'].unique(), name='model_name' )

            this_Z = []
            # for each model
            for mn in tdf['model_name'].unique():
                mdf =  tdf[( (tdf['model_name']==mn) &
                            (tdf['epoch']==best_val_epoch) )].drop_duplicates( 'ID' )
                
                Z = np.array( mdf['correct'] )
                this_Z.append( Z )


            # local updates
            exam_res_d['IDs'] = these_IDs
            exam_res_d['Z'] = np.array( this_Z )
            exam_res_d['model_names'] = these_mns


            print(f'writing {svar}_{full_split}_exam_res_d.pkl...')
            with open( f'exam_results/{svar}_{full_split}_exam_res_d.pkl', 'wb' ) as f:
                pickle.dump(exam_res_d, f, protocol=pickle.HIGHEST_PROTOCOL)



def create_jsonlines_dataset(SVAR, SPLIT):
    # NOTE; these come from PostProcessing.ipynb (?) -- disputed
    with open(f'exam_results/{SVAR}_{SPLIT}_exam_res_d.pkl', 'rb') as f:
        exam_res_d = pickle.load(f)

    these_mns = exam_res_d['model_names']
    # filter out LLMs for now
    these_mns = [ mn for mn in these_mns if (('Mistral' not in mn) and ('Llama2' not in mn))  ]
    this_Z = exam_res_d['Z']
    these_ids = exam_res_d['IDs']

    rows = []
    for sdx, subj in enumerate( these_mns ):
        resp_d = { cdx: int(this_Z[sdx, cdx]) for cdx in range(this_Z.shape[1]) }
        rows.append( {"subject_id": subj, "responses": resp_d} )

    write_jsonlines( f'py-irt/{SVAR}_{SPLIT}_irt_dataset.jsonlines', rows )

    dataset = py_irt.dataset.Dataset.from_jsonlines( f"py-irt/{SVAR}_{SPLIT}_irt_dataset.jsonlines" )

    return dataset, these_ids, these_mns


def run_1pl_model(dataset, num_epochs=500):
    config = IrtConfig(model_type='1pl', log_every=500, dropout=.2)
    trainer = IrtModelTrainer(config=config, data_path=None, dataset=dataset)
    trainer.train(epochs=num_epochs, device='cpu')

    return trainer


def get_irt_difficulty_ability(svar, split, num_epochs=500):
    dataset, these_ids, these_mns = create_jsonlines_dataset(svar, split)
    trainer = run_1pl_model(dataset, num_epochs=num_epochs)

    diff_df = pd.concat([ pd.Series(these_ids, name='ID'),
                        pd.Series(trainer.best_params['diff'], name='irt_difficulty') ], axis=1)
    diff_df['score_var'] = svar
    diff_df['split'] = split
    diff_df.to_csv( f"py-irt/{svar}_{split}_id_difficulty.csv", index=False )

    ability_df = pd.concat([ pd.Series(these_mns, name='model_name'),
                    pd.Series(trainer.best_params['ability'], name='irt_ability') ], axis=1)
    ability_df.to_csv( f"py-irt/{svar}_{split}_mn_ability.csv", index=False )

    return diff_df, ability_df






from sklearn.metrics import roc_auc_score, accuracy_score
import pickle
import pandas as pd

import sys
import numpy as np
from pydantic import BaseModel

import random
import py_irt
from py_irt.io import write_jsonlines
from py_irt.dataset import Dataset

from py_irt.config import IrtConfig
from py_irt.training import IrtModelTrainer


from utils.data_loading import *


ledger_len = pd.read_csv( 'data/ledger_len.csv' )


# WARNING; takes ~2min to run IRT on all splits
split_lst = ['train-None', 'train-Constant', 'val', 'test']
svar_lst =  ['SubjectiveLit', 'Numeracy', 'Anxiety', 'TrustPhys', 'wer']

# WARNING; takes ~1min
write_exam_results(split_lst, svar_lst, ledger_len)


# loop thru, fit py-irt models and write difficulty .csvs to py-irt/ folder
diff_df_lst = []
ab_df_lst = []
for split in split_lst:
    for svar in svar_lst:
        diff_df, ability_df = get_irt_difficulty_ability(svar=svar, split=split)
        ab_df_lst.append( ability_df )
        diff_df_lst.append( diff_df )


