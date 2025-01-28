import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score


def add_wer_demographics( this_ledger_df ):
    # add demographics to WER
    sub_wer = this_ledger_df[ this_ledger_df['score_var']=='wer' ]
    ledger_no_wer = this_ledger_df[ this_ledger_df['score_var']!='wer' ]

    wer_demos = pd.read_excel( 'data/data_zda/DenamedDemographics_and_more.xlsx' )
    sub_wer = pd.merge( sub_wer, wer_demos, how='left', left_on='user', right_on='Subject.ID' )

    return pd.concat( [ledger_no_wer, sub_wer] )


def add_prot_cols(tdf):
    this_svar = tdf['score_var'].iloc[0]

    if this_svar == 'wer':
        tdf['Sex'] = np.where( tdf['Subject.Gender']!='Male', 2, 1 )
        tdf['Age_Senior'] = np.where( tdf['Age'].astype(float)>=64, 1, 0 )
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

    # return ( priv_df['preds'].mean() / (prot_df['preds'].mean()+alpha) )
    return ( prot_df['preds'].mean() / (priv_df['preds'].mean()+alpha) )


def calculate_adj_disp_impact(tdf, this_demo, alpha=0.05):
    prot_df = tdf[tdf[this_demo]==1]
    priv_df = tdf[tdf[this_demo]!=1]

    p_prot = prot_df['preds'].mean()
    base_prot = prot_df['labels'].mean()

    p_priv = priv_df['preds'].mean()
    base_priv = priv_df['labels'].mean()

    return ( p_prot*base_priv) / ( ( p_priv*base_prot ) + alpha)


def proc_test_perform_row(tdf):
    tdf_full = add_prot_cols(tdf)

    this_acc = tdf_full['correct'].mean()
    this_auc = roc_auc_score( y_true=tdf_full['labels'], y_score=tdf_full['probs'] )
    this_f1 = f1_score( y_true=tdf_full['labels'], y_pred=tdf_full['preds'] )

    new_row = pd.DataFrame.from_dict( [{
        'model_name': tdf_full['model_name'].iloc[0], 'test_acc': this_acc,
        'test_auc': this_auc, 'test_f1': this_f1,
        'best_epoch': tdf_full['best_epoch'].iloc[0]
    }] )

    these_demos = ['Sex', 'Age_Senior', 'Race_POC', 'Education_Low', 'Income_Low', 'ESL']
    for this_demo in these_demos:
        new_row[this_demo+"|DI"] = calculate_disp_impact(tdf_full, this_demo)
        new_row[this_demo+"|ADI"] = calculate_adj_disp_impact(tdf_full, this_demo)

    return new_row


# NOTE; takes 10min and n-1 cores
# we save a checkpoint of the data
def create_test_perform_full():

    import os
    this_dir = os.getcwd()
    os.chdir('..')
    os.chdir('..')

    print("> loading ledger -- this will take 3min...")
    
    ledger_diff = pd.read_csv( '/Users/ryancook/Downloads/ledger_len.csv' )
    ledger_diff['strat'] = [ s[-1] for s in ledger_diff['model_name'].str.split('-') ]

    # add WER demographics (since we got these late)
    ledger_diff = add_wer_demographics( ledger_diff )

    def create_perform_tdf(mn):
        this_log = pd.read_csv( f'outputs/logs3/{mn}_log.txt', sep='\t' )
        this_best_ep = this_log.loc[this_log['val_auc'].idxmax(), 'epoch']
        tdf = ledger_diff[( (ledger_diff['model_name']==mn) &
                        (ledger_diff['epoch']==this_best_ep) &
                        (ledger_diff['split_full']=='test') )]
        tdf['best_epoch'] = this_best_ep
        return tdf



    these_mns = list( ledger_diff['model_name'].unique() )

    import multiprocess
    #multisetup
    cores_to_use = multiprocess.cpu_count()-1
    pool = multiprocess.Pool(cores_to_use)

    # multi create
    # WARNING; takes ~1min
    print("> processing results files -- this will take 1min...")
    with multiprocess.Pool(cores_to_use) as pool:
        tdf_lst = pool.map(create_perform_tdf, these_mns)

    # multi proc
    with multiprocess.Pool(cores_to_use) as pool:
        perform_lst = pool.map(proc_test_perform_row, tdf_lst)


    mt_d = {
        'bert-base-uncased': 'bert',
        'xlm-roberta-base-uncased-all-english': 'roberta',
        'local-ffn': 'ffn',
        'local-cnn': 'cnn',
        'local-lstm': 'lstm'
    }

    # concat and clean the performance dataframe
    test_perform_df = pd.concat( perform_lst ).reset_index(drop=True)

    test_perform_df['score_var'] = [ s.split('_')[1] for s in test_perform_df['model_name'] ]
    test_perform_df['strat'] = [ s.split('_')[-1].split('-')[1] for s in test_perform_df['model_name'] ]
    test_perform_df['model_type'] = [ mt_d[ s[0] ] for s in test_perform_df['model_name'].str.split('_') ]


    # add ability
    all_ab_df = pd.DataFrame()
    for svar in test_perform_df['score_var'].unique():
        ab_df = pd.read_csv( f"py-irt/{svar}_test_mn_ability.csv" )
        all_ab_df = pd.concat([ all_ab_df, ab_df ])


    # merge
    test_perform_full = pd.merge( test_perform_df, all_ab_df, on='model_name', how='left' )

    id_cols = ['model_name', 'model_type', 'score_var', 'strat', 'best_epoch']
    perform_cols = [ 'test_acc', 'test_auc', 'test_f1', 'irt_ability' ]
    di_cols = [ c for c in test_perform_full if '|DI' in c ]
    adi_cols = [ c for c in test_perform_full if '|ADI' in c ]
    test_perform_full = test_perform_full[id_cols+perform_cols+di_cols+adi_cols]

    os.chdir(this_dir)
    return test_perform_full





def create_cormat(in_df, comp_cols):
    out_mat = in_df.groupby( ['score_var', 'model_name'] )[ comp_cols ].corr(method='spearman').reset_index().rename(columns={'level_2': 'metric'})
    out_mat['strat'] = [ s.split('_')[-1].split('-')[1] for s in out_mat['model_name'] ]

    return out_mat


def calculate_micro_avg(in_mat, comp_cols):
    out_gdf = in_mat.groupby( 'metric' )[ comp_cols ].agg(['mean', 'std', 'count']).reset_index().rename(columns={'level_2': 'metric'})
    out_gdf.columns = [ out_gdf.columns.get_level_values(0)[cdx] + '_' + c if len(c)>0 else out_gdf.columns.get_level_values(0)[cdx]
                     for cdx, c in enumerate(out_gdf.columns.get_level_values(1)) ]
    
    return out_gdf


def create_agg_corr_df(in_df, comp_cols):
    corr_df = pd.DataFrame()

    for m1 in sorted(comp_cols):
        for m2 in sorted(comp_cols):
            # skip if equal or we've already calculated
            if m1==m2: continue
            if len(corr_df)>0:
                if f'{m2} : {m1}' in list(corr_df['metrics']): continue

            # print(f'{m1}-{m2}')
            corr_mean = in_df[in_df['metric']==m1][f'{m2}_mean'].iloc[0]
            corr_std = in_df[in_df['metric']==m1][f'{m2}_std'].iloc[0]
            count_model = in_df[in_df['metric']==m1][f'{m2}_count'].iloc[0]
            new_row = pd.DataFrame.from_dict([{ 
                'metrics': f'{m1} : {m2}', 'mean_scorr': corr_mean,
                'std_scorr': corr_std, 'count_models': count_model,
            }])

            corr_df = pd.concat([ corr_df, new_row ])

    return corr_df.reset_index(drop=True)