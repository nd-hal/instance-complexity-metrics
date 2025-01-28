import os
import numpy as np
import pandas as pd



def remove_duplicate_id(this_id, this_ledger):
    rep_df = this_ledger[( (this_ledger['ID']!=this_id) &
                        (this_ledger['score_var']=='Anxiety') &
                        (this_ledger['split']=='test') )]

    rm_df = this_ledger[( (this_ledger['score_var']=='Anxiety') &
                        (this_ledger['split']=='test') )]

    oth_df = this_ledger[( (this_ledger['score_var']!='Anxiety') |
                        (this_ledger['split']!='test') )]

    # concat and sort / reset the index
    new_ledger = pd.concat([ oth_df, rep_df ]).sort_index().reset_index(drop=True)

    return new_ledger


def load_and_clean_ledgers():
    # WARNING: takes ~2min to load 15M rows and process
    this_dir = 'ledgers/'
    all_ledgers = pd.DataFrame()
    for fdx, file in enumerate( os.listdir(this_dir) ):
        if file[0] != '.':
            if fdx%10==0:
                print(f" {fdx} / {len(os.listdir(this_dir))} ")
            tdf = pd.read_csv( this_dir+file, keep_default_na=False )

            if 'text' in tdf.columns:
                tdf = tdf.drop('text', axis=1)
                tdf.to_csv( this_dir+file , index=False)

            # # NOTE; filtering to only TEST for efficiency
            # tdf = tdf[tdf['split']=='test']
            all_ledgers = pd.concat([ all_ledgers, tdf])


    print('cleaning...')
    # clean up issue from local processing
    all_ledgers = all_ledgers.iloc[:, :9]
    all_ledgers = all_ledgers.reset_index(drop=True)
    # # loop thru names once to get info on task and model type
    # mt_sv_lst = [ (s[0], s[1]) for s in all_ledgers['model_name'].str.split('_') ]
    all_ledgers['score_var'] = [ s[1] for s in all_ledgers['model_name'].str.split('_') ]
    # all_ledgers['model_type'] = [ t[0] for t in mt_sv_lst ]
    all_ledgers['strat'] = [ s[-1] for s in all_ledgers['model_name'].str.split('-') ]

    # remove a problem id from a bad join on Anxiety-test
    # WARNING; takes ~5min
    PROBLEM_ID = 'f5_r594'
    all_ledgers = remove_duplicate_id(PROBLEM_ID, all_ledgers)

    all_ledgers['split_full'] = [ s + '-' + all_ledgers['strat'].iloc[sdx] if s=='train' else s for sdx, s in enumerate(all_ledgers['split']) ]

    return new_ledgers






## PYHARD ##

# need to create train predictions (since ledger only stores val and test)

# modified to work with df instead of dataframe
def write_predictions(model_name, test_probs, tdf, split_full):
    tdf = tdf.reset_index(drop=True)
    with open(f"predictions/{split_full}/{model_name}_predictions.txt", "w") as predictionsFile:
        predictionsFile.write("Id|Text|Prediction\n")
        for i in range(0, len(test_probs)):
            # ensure we have the text too
            predictionsFile.write(str(tdf.loc[i, 'ID']) + "|" + str(tdf.loc[i, 'text']) + "|" + str(test_probs[i]) + "\n")

def create_model_responses(ledger_hardness):
    sub_ledger = ledger_hardness[['model_name', 'epoch', 'split_full', 'ID', 'probs', 'score_var']]

    for svar in ledger_hardness['score_var'].unique():
        sv_ledger = sub_ledger[sub_ledger['score_var']==svar]

        if svar == 'wer':
                print('using ZDA data')
                input_data_d = load_zda_data('data/data_zda/', subset=False)
        else:
            print('using HAL data')
            input_data_d = load_hal_data('data/DataCVFolds/', score_variable=svar, subset=False)

        input_data_d['hard_train_data'], input_data_d['rand_train_data'] = sample_hard_rand(input_data_d['train_data'], svar)
        
        for mn in sv_ledger['model_name'].unique():
            # print(mn)
            this_log = pd.read_csv( f'logs/{mn}_log.txt', sep='\t' )
            this_best_ep = this_log.loc[this_log['val_auc'].idxmax(), 'epoch']
            tdf = sv_ledger[( (sv_ledger['model_name']==mn) &
                            (sv_ledger['epoch']==this_best_ep) )]
            
            if 'Constant' in mn:
                tr_join = pd.merge( tdf[tdf['split_full']=='train-Constant'], input_data_d['hard_train_data'],
                                    on='ID', how='left' )
                write_predictions( mn, list( tdf[tdf['split_full']=='train-Constant']['probs'] ), tr_join, 'train-Constant' )
            else:
                tr_join = pd.merge( tdf[tdf['split_full']=='train-None'], input_data_d['rand_train_data'],
                                    on='ID', how='left' )
                write_predictions( mn, list( tdf[tdf['split_full']=='train-None']['probs'] ), tr_join, 'train-None' )

            
            val_join = pd.merge( tdf[tdf['split_full']=='val'], input_data_d['val_data'],
                            on='ID', how='left' )
            write_predictions( mn, list( tdf[tdf['split_full']=='val']['probs'] ), val_join, 'val' )


def get_full_split_df(svar, full_split):
    if svar == 'wer':
        # print('using ZDA data')
        input_data_d = load_zda_data('data/data_zda/', subset=False)
    else:
        # print('using HAL data')
        input_data_d = load_hal_data('data/DataCVFolds/', score_variable=svar, subset=False)

    # print(f'{svar}-{full_split}')
    short_split = full_split.split('-')[0]
    tdf = input_data_d[f'{short_split}_data']
    if svar=='Anxiety' and full_split=='test':
        PROBLEM_ID = 'f5_r594'
        tdf = tdf[tdf['ID']!=PROBLEM_ID].reset_index(drop=True)
    
    # need to subset train
    if 'train' in full_split:
        sdf = pd.read_csv( f'splits/{svar}_train_ids.csv' ).iloc[:, 1:]
        if 'None' in full_split:
            tdf = tdf[ tdf['ID'].isin(sdf['rand_ids']) ]
        else:
            tdf = tdf[ tdf['ID'].isin(sdf['hard_ids']) ]

    return tdf


def create_model_corr_df(in_df):
    corr_df = pd.DataFrame()
    this_mn = in_df['model_name'].iloc[0]
    this_svar = in_df['score_var'].iloc[0]
    this_strat = in_df['strat'].iloc[0]

    for m1 in sorted(comp_cols):
        for m2 in sorted(comp_cols):
            # skip if equal or we've already calculated
            if m1==m2: continue
            if len(corr_df)>0:
                if f'{m2} : {m1}' in list(corr_df['metrics']): continue

            # print(f'{m1}-{m2}')
            this_corr = in_df[in_df['metric']==m1][f'{m2}'].iloc[0]
            new_row = pd.DataFrame.from_dict([{ 
                'score_var': this_svar, 'model_name': this_mn,
                'strat': this_strat,
                'metrics': f'{m1} : {m2}', 'scorr': this_corr
            }])

            corr_df = pd.concat([ corr_df, new_row ])

    return corr_df.reset_index(drop=True)



def add_prot_cols(tdf):
    this_svar = tdf['score_var'].iloc[0]

    if this_svar == 'wer':
        tdf['Sex'] = np.where( tdf['Subject.Gender']!='Male', 2, 1 )
        tdf['Age_Senior'] = np.where( tdf['Age_y'] >= 50, 1, 0 )     # 33 yrs old is median
        tdf['Race_POC'] = np.where( tdf['Subject.Race']!='White/Caucasian', 1, 0 )
        tdf['Education_Low'] = np.where( ((tdf['Subject.Education.Level']=='Less Than High School') | 
                                                (tdf['Subject.Education.Level']=='College or Trade or Vocational School')), 1, 0 )
        # tdf['Income_Low'] = np.where( tdf['Income_Cat']<3, 1, 0 )
        tdf['Income_Low'] = None      # not provided
        tdf['ESL'] = 1        # all participants must speak English as first language

    else:
        tdf['Age_Senior'] = np.where( tdf['Age']>=65, 1, 0 )
        tdf['Race_POC'] = np.where( tdf['Race']!=1, 1, 0 )
        tdf['Education_Low'] = np.where( tdf['Education']<3, 1, 0 )
        tdf['Income_Low'] = np.where( tdf['Income_Cat']<3, 1, 0 )
        tdf['ESL'] = np.where( tdf['English_First_Lang']!=1, 1, 0 )

    return tdf


def calculate_disp_impact(tdf, this_demo, alpha=0.005):
    prot_df = tdf[tdf[this_demo]==1]
    priv_df = tdf[tdf[this_demo]!=1]

    return ( prot_df['preds'].mean() / (priv_df['preds'].mean()+alpha) )


def calculate_adj_disp_impact(tdf, this_demo, alpha=0.05):
    prot_df = tdf[tdf[this_demo]==1]
    priv_df = tdf[tdf[this_demo]!=1]

    p_prot = prot_df['preds'].mean()
    base_prot = prot_df['labels'].mean()

    p_priv = priv_df['preds'].mean()
    base_priv = priv_df['labels'].mean()

    return ( p_prot*base_priv) / ( ( p_priv*base_prot ) + alpha)