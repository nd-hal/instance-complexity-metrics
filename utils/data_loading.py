import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset

# from utils.data_processing import *


# used for efficient testing on subsets
def stratified_sample(df, n):
    return pd.concat( [ df[df.label!=1].sample(n=round(n/2), random_state=123),
                df[df.label==1].sample(n=round(n/2), random_state=123) ] ).sample(
                    frac=1, random_state=123).reset_index(drop=True)


###################################################################
################### Loading Data from Local Files #################
###################################################################
# HAL PSYCHOMETRIC NLP

def load_and_process_demo(directory, score_variable):
    id_table = pd.read_csv(directory+'Data_SurveyPlusDemographics.txt', sep='\t', low_memory=False)

    cmap_d = {
        'D1': 'Age',
        'D2': 'Sex',
        'D3': 'Race',
        'D4': 'Education',
        'D5': 'Income_Cat',
        'D6': 'English_First_Lang',
        'Dmed_7': 'Height_In',
        'Dmed_8': 'Weight_Lb',
        'DMed_1': 'Prescriptions_Num',
        'DMed_2': 'Has_Primary_Care_Phys',
        'DMed_3': 'Num_Visits_to_Phys_Two_Year',
        'DMed_9': 'Hours_Exercise_Week',
        'Dmed_10': 'Eating_Health_Cat',
        'Dmed_5': 'Smoke_Num_Week',
        'Dmed_6': 'Alc_Num_Week'
    }

    id_table = id_table.rename(columns=cmap_d)
    c_lst= [ 'File', 'Row', 'Text_'+score_variable ] + list(cmap_d.values())

    return id_table[c_lst]

# parent function for calling loading modules
# note the score variable is the DV and continuous variable used for median split
def load_hal_data(directory, score_variable, subset=False):
    med_input_data_d = load_files(directory+'MedianCV/1', score_variable)
    cont_input_data_d = load_files(directory+'ContinuousCV/1', score_variable)
    # merge the score
    input_data_d = {}
    for key in med_input_data_d:
        tdf = med_input_data_d[key].reset_index(drop=True)
        # tdf['id'] = key.split('_')[0] + '_' + tdf.index.astype(str)
        scoreCol = cont_input_data_d[key][['label']].rename(columns={'label': 'score'})

        # join demographic info
        id_table = load_and_process_demo(directory, score_variable)
        # fix a problem in the Anxiety test set from duplicate text
        if key == 'test_data':
            id_table = id_table[id_table['File']==1].reset_index(drop=True)

        jdf = pd.merge(tdf, id_table, left_on='text',
                        right_on='Text_'+score_variable, how='left').drop('Text_'+score_variable, axis=1)
        id_Ser = pd.Series( ( 'f'+ jdf['File'].astype(str) + '_r' + jdf['Row'].astype(str) ), name='ID' )

        # fix a problem in the Anxiety test set from duplicate text
        # NOTE; 'File' column refers to the fold in which this person is in the test set
        if key == 'test_data':
            jdf = jdf[jdf['File']==1].reset_index(drop=True)

        jdf = pd.concat([ id_Ser, jdf ], axis=1)
        
        input_data_d[key] = pd.concat([jdf, scoreCol], axis=1)

    if subset:
        input_data_d = {
            'train_data': stratified_sample(input_data_d['train_data'].dropna().reset_index(drop=True), 200),
            'val_data': stratified_sample(input_data_d['val_data'].dropna().reset_index(drop=True), 100),
            'test_data': stratified_sample(input_data_d['test_data'].dropna().reset_index(drop=True), 100)
        }
        
    return input_data_d

# loading module to loop thru directory and extract files
def load_files(directory, score_variable):
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    # for folder in os.listdir(directory):
    #     if '.' not in folder:
    this_dir = os.listdir(directory)
    for filename in sorted(this_dir):
        if score_variable in filename:
            this_df = pd.read_csv('/'.join([directory, filename]),
                                delimiter='\t', header=None)
            this_df.columns = ['label', 'text']
            # this_df['test_folder'] = folder

            if 'train' in filename:
                train_data = pd.concat([train_data, this_df])
            elif 'val' in filename:
                val_data = pd.concat([val_data, this_df])
            else:
                test_data = pd.concat([test_data, this_df])

    return {
        'train_data': train_data.dropna().reset_index(drop=True),
        'val_data': val_data.dropna().reset_index(drop=True),
        'test_data': test_data.dropna().reset_index(drop=True)
    }


# ADVERSE DRUG DETECTION
# only one function needed to extract files
def load_drug_data(directory, subset=False):
    train_data = pd.read_csv(directory+'TextFiles/train_DrugExp_Text.tsv', sep='\t', header=None)
    train_data.columns = ['label', 'text']
    train_data['score'] = 0
    val_data = pd.read_csv(directory+'TextFiles/validation_DrugExp_Text.tsv', sep='\t', header=None)
    val_data.columns = ['label', 'text']
    val_data['score'] = 0
    test_data = pd.read_csv(directory+'TextFiles/test_DrugExp_Text.tsv', sep='\t', header=None)
    test_data.columns = ['label', 'text']
    test_data['score'] = 0

    if subset:
        return {
            'train_data': stratified_sample(train_data.dropna().reset_index(drop=True), 200),
            'val_data': stratified_sample(val_data.dropna().reset_index(drop=True), 100),
            'test_data': stratified_sample(test_data.dropna().reset_index(drop=True), 100)
        }
    else:
        return {
            'train_data': train_data.dropna().reset_index(drop=True),
            'val_data': val_data.dropna().reset_index(drop=True),
            'test_data': test_data.dropna().reset_index(drop=True)
        }


# ZOOM DEPRESSION DATA
# needs a lot of helper functions
# maps users to MDD/Control
def get_id_labs(label_df):
    tdf = (label_df.iloc[:, 1:].isna()==False).astype(int)

    these_labs = []
    for idx in tdf.index:
        this_row = tdf.loc[idx, :]
        if this_row.sum()>0:
            # print(idx)
            these_labs.append(this_row.index[this_row==1][0])
        else:
            these_labs.append(pd.NA)

    udf = pd.concat([label_df['Patient ID'], pd.Series(these_labs, name='label')], axis=1)
    udf = udf.dropna().reset_index(drop=True)
    id_labs = udf.iloc[:, [0, -1]]

    return id_labs

# merges the labels and removes Other class
def add_labels_and_subset(df, label_df):
    id_labs = get_id_labs(label_df)
    join_df = pd.merge( df, id_labs, how='left', left_on='user', right_on='Patient ID' )

    sub_df = join_df[join_df['label']!='Other'].reset_index(drop=True)
    sub_df['label_num'] = np.where(sub_df['label']=='MDD', 1, 0)
    sub_df = sub_df.drop('label', axis=1)
    sub_df = sub_df.rename(columns={'label_num': 'label', 'content': 'text'})

    return sub_df

# edit distance between the AWS and Gold (ground truth)
def score_edit_distance(aws_df, gold_df, both_users):
    import editdistance

    ueds = []
    for user in both_users:
        adf = aws_df[aws_df['user']==user]
        gdf = gold_df[gold_df['user']==user]

        gstr = ' '.join(gdf['text'])
        astr = ' '.join(adf['text'])

        ueds.append( editdistance.eval(gstr, astr) / len(gstr) )

    id_score = pd.concat([ pd.Series(gold_df['user'].unique(), name='user'),
            pd.Series(ueds, name='score') ], axis=1)
    
    return id_score

# need a custom function here to stratify sample on users (not documents)
def train_val_test_split(these_users):
    N_users = len(these_users)
    np.random.seed(123)
    np.random.shuffle(these_users)

    # NOTE; 60% of users in train, 20% in val, (20% in test)
    N_tr = round( N_users*.6 )
    N_val = round( N_users*.2 )

    tr_users = these_users[:N_tr]
    v_users = these_users[N_tr:N_tr+N_val]
    te_users = these_users[N_tr+N_val:]

    return tr_users, v_users, te_users

# parent function to load the data
def load_zda_data(directory, subset=False):
    aws = pd.read_csv(directory + 'awsSingleTextRolled.csv')
    aws_psd_oth = aws[aws['interview_part'].isin(pd.Series(['Pre', 'Semi', 'Demo']))].reset_index(drop=True)
    gold = pd.read_csv(directory + 'goldSingleText.csv')
    gold_psd_oth = gold[gold['interview_part'].isin(pd.Series(['Pre', 'Semi', 'Demo']))].reset_index(drop=True)

    dn_labels = pd.read_excel('data/data_zda/De-Name-Labels.xlsx')
    dn_labels = dn_labels.loc[1:,].reset_index(drop=True)

    sub_aws = add_labels_and_subset(aws_psd_oth, dn_labels)
    sub_gold = add_labels_and_subset(gold_psd_oth, dn_labels)


    all_users = list(set( list(sub_gold['user'].unique()) + list(sub_aws['user'].unique()) ))
    both_users = [ u for u in all_users if u in list(sub_gold['user']) and u in list(sub_aws['user']) ]

    sub_aws = sub_aws[sub_aws['user'].isin(pd.Series(both_users))].reset_index(drop=True)
    sub_gold = sub_gold[sub_gold['user'].isin(pd.Series(both_users))].reset_index(drop=True)

    id_score = score_edit_distance(sub_aws, sub_gold, both_users)

    sub_aws_scored = pd.merge(sub_aws, id_score, how='left', on='user')

    # add an ID column
    sub_aws_scored['ID'] = 'u' + sub_aws_scored['user'] + '_r' + sub_aws_scored.index.astype(str)

    # add demographics
    these_demos = pd.read_excel( 'data/data_zda/DenamedDemographics_and_more.xlsx' )
    sub_aws_scored = pd.merge( sub_aws_scored, these_demos, how='left',
                              left_on='user', right_on='Subject.ID' )

    # train-test split
    ctrl_users = sub_aws_scored[sub_aws_scored['label']!=1]['user'].unique()
    mdd_users = sub_aws_scored[sub_aws_scored['label']==1]['user'].unique()

    ctrl_tr_users, ctrl_v_users, ctrl_te_users = train_val_test_split(ctrl_users)
    mdd_tr_users, mdd_v_users, mdd_te_users = train_val_test_split(mdd_users)

    train_users =pd.Series( list(ctrl_tr_users) + list(mdd_tr_users) )
    val_users = pd.Series( list(ctrl_v_users) + list(mdd_v_users) )
    test_users = pd.Series( list(ctrl_te_users) + list(mdd_te_users) )

    sub_cols = ['user', 'text', 'interview_part', 'Patient ID', 'label', 'score', 'ID']
    train_data = sub_aws_scored[sub_aws_scored['user'].isin(train_users)].dropna(subset=sub_cols).reset_index(drop=True)
    val_data = sub_aws_scored[sub_aws_scored['user'].isin(val_users)].dropna(subset=sub_cols).reset_index(drop=True)
    test_data = sub_aws_scored[sub_aws_scored['user'].isin(test_users)].dropna(subset=sub_cols).reset_index(drop=True)

    if subset:
        return {
            'train_data': stratified_sample(train_data.dropna(subset=sub_cols).reset_index(drop=True), 200),
            'val_data': stratified_sample(val_data.dropna(subset=sub_cols).reset_index(drop=True), 100),
            'test_data': stratified_sample(test_data.dropna(subset=sub_cols).reset_index(drop=True), 100)
        }
    else:
        return {
            'train_data': train_data.dropna(subset=sub_cols).reset_index(drop=True),
            'val_data': val_data.dropna(subset=sub_cols).reset_index(drop=True),
            'test_data': test_data.dropna(subset=sub_cols).reset_index(drop=True)
        }



def create_hard_d( svar, subset=False ):
    these_splits = [ 'train', 'val', 'test' ]

    data_hard_d = {}
    input_data_d = load_hal_data('data/DataCVFolds/', score_variable=svar, subset=False)

    for split in these_splits:
        # print(f'{svar}-{split}')
        this_input_data = input_data_d[split+'_data']
        
        full_path = 'pyhard/' + f'{svar}/{split}/'
        this_ih = pd.read_csv( full_path + 'ih.csv' )

        if len(this_ih) != len(this_input_data):
            raise ValueError('Shape of input data must match instance hardness calculations')

        ih_hard = pd.concat([ this_input_data, this_ih ], axis=1)
        ih_hard['score_var'] = svar
        ih_hard['split'] = split
        ih_hard = stratified_sample(ih_hard.dropna().reset_index(drop=True), 200) if subset else ih_hard

        data_hard_d[ split+'_data' ] = ih_hard

    return data_hard_d



def get_bert_embs( svar ):
    these_splits = [ 'train', 'val', 'test' ]

    embs_d = {}
    hard_d = create_hard_d( svar )

    for split in these_splits:
        # print(f'{svar}-{split}')
        this_input_data = hard_d[split+'_data']
        
        full_path = 'pyhard/' + f'{svar}/{split}/'
        this_embs = pd.read_csv( full_path + 'data.csv' ).values

        ID_to_edx = { this_input_data['ID'].iloc[edx]: edx for edx in range(len(this_embs)) }
        edx_to_ID = { edx: this_input_data['ID'].iloc[edx] for edx in range(len(this_embs)) }

        if len(this_embs) != len(this_input_data):
            raise ValueError('Shape of input data must match BERT embeddings')

        embs_d[ split+'_data' ] = {}
        embs_d[ split+'_data' ][ 'ID' ] = list( ID_to_edx.keys() )
        embs_d[ split+'_data' ][ 'embs' ] = this_embs[:, :-1]
        embs_d[ split+'_data' ][ 'label' ] = this_embs[:, -1]
        embs_d[ split+'_data' ][ 'score' ] = list( this_input_data['score'] )
        embs_d[ split+'_data' ][ 'ih' ] = list( this_input_data['instance_hardness'] )

    return embs_d, ID_to_edx, edx_to_ID



# Create a custom Dataset class for your data
class CustomDataset(Dataset):
    def __init__(self, embs_d):

        self.ID = embs_d[ 'ID' ]
        self.embs = embs_d[ 'embs' ]
        self.label = embs_d[ 'label' ]
        self.score = embs_d[ 'score' ]
        self.ih = embs_d[ 'ih' ]

    def __len__(self):
        return len(self.ID)

    def __getitem__(self, idx):
        return {
            'ID': self.ID[idx],
            'embs': self.embs[idx, :],
            'label': self.label[idx],
            'score': self.score[idx],
            'ih': self.ih[idx],
        }


def create_loaders(embs_d, batch_size, shuffle=False):
    these_splits = [ 'train', 'val', 'test' ]
    load_d = {}
    for split in these_splits:

        this_set = CustomDataset( embs_d[split+'_data'] )
        this_loader = DataLoader( this_set, batch_size=batch_size, shuffle=shuffle )

        load_d[ split ] = this_loader

    return load_d