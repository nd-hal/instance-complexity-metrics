import itertools
import pickle
import sys
import keras

import torch
from torch.utils.data import DataLoader, Dataset

from nltk.tokenize import word_tokenize

from utils.data_loading import *
from utils.data_processing import *
from utils.transformer_functions import *
from utils.input_output import *

from utils.ffn_functions import *
from utils.lstm_functions import *
from utils.cnn_functions import *


# get args
function_class, score_var, rev, offset, finetune, subset = get_inputs_from_args()

# function_class = 'llama'
# score_var = 'TrustPhys'
# rev = False
# offset = 0
# finetune = True
# subset = True
rev, offset, all_param_d, grid_keys, all_param_lst, model_param_d = setup_params(function_class, score_var, rev, offset, finetune).values()


access_token = '<INSERT ACCESS TOKEN HERE>'

# allow us to run backwards
if rev:
    all_param_lst = all_param_lst[::-1]

# loop thru models / param sets
for tpdx, tup in enumerate(all_param_lst[offset:]):
    # need to flip the offset on reverse
    this_offset = tpdx+offset if not rev else len(all_param_lst)-(tpdx+offset)
    these_params = dict(zip(list(all_param_d.keys()), tup))
    print(these_params)
    these_model_params = { k:v for k,v in these_params.items() if k in model_param_d }

    # binary variable for using Pre-trained language model (i.e., transformer) or local
    is_plm = 'local' not in these_params['modelCheckpoint']


    # LOAD AND PREPARE DATA
    # 'scoreVariable': ['SubjectiveLit', 'Numeracy', 'Anxiety', 'TrustPhys', 'WER']
    if score_var == 'wer':
        print('using ZDA data')
        input_data_d = load_zda_data('data/data_zda/', subset=subset)
    elif score_var == 'drug':
        print('using DRUG data')
        input_data_d = load_drug_data('data/', subset=subset)
    else:
        print('using HAL data')
        input_data_d = load_hal_data('data/DataCVFolds/', score_variable=these_params['scoreVariable'], subset=subset)

    # create hard, random subsets
    input_data_d['hard_train_data'], input_data_d['rand_train_data'] = sample_hard_rand(input_data_d['train_data'], score_var)
    
    
    # prepare data
    tokenizer = setup_tokenizer(these_params['modelCheckpoint'], access_token) if is_plm else word_tokenize
    hard_train_dataset, rand_train_dataset, val_dataset, test_dataset = prepare_data(input_data_d, tokenizer,
                                                                                     these_params['embType'], these_params['maxLength']).values()
    # set initial train_dataset from hard or random
    train_dataset = hard_train_dataset if use_hard_data(these_params['difficultTime'], 0) else rand_train_dataset

    # model setup
    if is_plm:
        # transformers need dataloaders
        train_dataloader, val_dataloader, test_dataloader = set_data_loaders(these_params, 0, hard_train_dataset, rand_train_dataset,
                                                                         val_dataset, test_dataset).values()

        # Set up optimizer and scheduler
        model, device, optimizer, total_steps, scheduler = setup_transformer(train_dataset, train_dataloader, these_params['modelCheckpoint'],
                                                                             these_params['epochs'], access_token, **these_model_params)

    elif 'ffn' in these_params['modelCheckpoint']:
        # ffn
        model = createFFN(train_dataset, len(set(train_dataset.labels)), **these_model_params)
    elif 'cnn' in these_params['modelCheckpoint']:
        model = createWordCNN(train_dataset, len(set(train_dataset.labels)), **these_model_params)
    else:
        # lstm
        model = createWordLSTM(train_dataset, len(set(train_dataset.labels)), **these_model_params)
    
    model_name = setup_model_name(this_offset, these_params, grid_keys, these_params['modelCheckpoint'],
                                  these_params['scoreVariable'], finetune=finetune, subset=subset)
    print(model_name)


    # check if we've already trained this model
    resultsLocation = f"logs/{model_name}_log.txt"
    if os.path.exists(resultsLocation):
        print(f"%%%%%%%%%%%%%%%%%%%%%% SKIPPING: {model_name} %%%%%%%%%%%%%%%%%%%%%%")
        continue
    else:
        with open(f"logs/{model_name}_log.txt", "a") as outputFile:
            outputFile.write("epoch\ttrain_loss\ttrain_acc\ttrain_auc\tval_loss\tval_acc\tval_auc\n")

    # clear memory
    torch.cuda.empty_cache()

    train_losses = []       # needed for patience
    best_val_auc = 0
    MODEL_LEDGER = pd.DataFrame()
    # loop thru epochs
    for epoch in range(1, these_params['epochs']+1):
        print(f"Epoch {epoch} / {these_params['epochs']}")

        # fintetune a PLM
        if is_plm:
            # just using loaders from here out
            train_dataloader, val_dataloader, test_dataloader = set_data_loaders(these_params, epoch, hard_train_dataset, rand_train_dataset,
                                                                                val_dataset, test_dataset).values()
            # TRAINING
            if finetune:
                ( tr_loss, tr_acc, tr_auc, tr_probs, tr_preds,
                 tr_true, tr_losses, tr_IDs, tr_texts ) = train_transformer(model, device, optimizer, scheduler, train_dataloader, epoch, these_params['epochs'])
            else:
                ( tr_loss, tr_acc, tr_auc, tr_probs, tr_preds,
                 tr_true, tr_losses, tr_IDs, tr_texts ) = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, n.nan
            # VALIDATION
            ( val_loss, val_acc, val_auc, val_probs, val_preds,
             val_true, val_losses, val_IDs, val_texts ) = validate_transformer(model, device, val_dataloader, epoch, these_params['epochs'])

            # TESTING
            ( te_loss, te_acc, te_auc, te_probs, te_preds,
             te_true, te_losses, te_IDs, te_texts ) = validate_transformer(model, device, test_dataloader, epoch, these_params['epochs'])

        # train a local model
        else:
            # chose easy/hard train set -- happens under the hood in set_data_loaders() above
            train_dataset = hard_train_dataset if use_hard_data(these_params['difficultTime'], epoch) else rand_train_dataset

            # TRAINING
            fit_d = model.fit(train_dataset.pt_embeddings, train_dataset.labels, batch_size=these_params['batchSize'], epochs=1, verbose=0)
            tr_loss, tr_acc = fit_d.history['loss'][0], fit_d.history['accuracy'][0]
            tr_true = train_dataset.labels
            tr_probs, tr_auc, tr_preds = calculate_metrics(model, train_dataset.pt_embeddings, tr_true, is_val=False)
            tr_probs_duo = np.array([ [1-v, v] for v in tr_probs ] )    # need to reshape to calculate loss
            tr_losses = keras.losses.sparse_categorical_crossentropy(train_dataset.labels, tr_probs_duo).numpy()
            tr_IDs = train_dataset.ID
            tr_texts = train_dataset.texts

            # VALIDATION
            val_loss, val_acc, val_auc, val_probs, val_preds, val_true = calculate_metrics(model, val_dataset.pt_embeddings, val_dataset.labels, is_val=True)
            val_probs_duo = np.array([ [1-v, v] for v in val_probs ] )    # need to reshape to calculate loss
            val_losses = keras.losses.sparse_categorical_crossentropy(val_dataset.labels, val_probs_duo).numpy()
            val_IDs = val_dataset.ID
            val_texts = val_dataset.texts
            print(f"Val AUC: {val_auc}")

            # TESTING
            # treating like val to track everything for ledger
            te_loss, te_acc, te_auc, te_probs, te_preds, te_true = calculate_metrics(model, test_dataset.pt_embeddings, test_dataset.labels, is_val=True)
            te_probs_duo = np.array([ [1-v, v] for v in te_probs ] )    # need to reshape to calculate loss
            te_losses = keras.losses.sparse_categorical_crossentropy(test_dataset.labels, te_probs_duo).numpy()
            te_IDs = test_dataset.ID
            te_texts = test_dataset.texts


        # UPDATE LEDGER
        # this tracks EVERYTHING -- probs, preds, labels, losses, etc.
        tr_d = {}
        tr_d.update({'model_name': model_name, 'epoch': epoch, 'split': 'train',
                     'ID': tr_IDs, 'text': tr_texts, 'labels': tr_true,
                     'probs': tr_probs, 'preds': tr_preds, 'losses': tr_losses})
        val_d = {}
        val_d.update({'model_name': model_name, 'epoch': epoch, 'split': 'val',
                     'ID': val_IDs, 'text': val_texts, 'labels': val_true,
                     'probs': val_probs, 'preds': val_preds, 'losses': val_losses})
        te_d = {}
        te_d.update({'model_name': model_name, 'epoch': epoch, 'split': 'test',
                     'ID': te_IDs, 'text': te_texts, 'labels': te_true,
                     'probs': te_probs, 'preds': te_preds, 'losses': te_losses})
        
        MODEL_LEDGER = pd.concat([ MODEL_LEDGER, pd.DataFrame.from_dict(tr_d),
                                  pd.DataFrame.from_dict(val_d), pd.DataFrame.from_dict(te_d) ])
        # let's overwrite the ledger each epoch to be safe
        MODEL_LEDGER.to_csv(f'ledgers/{model_name}.csv', index=False)
        

        # WRITE LOGS / OUTPUTS
        # this is the log file used for progress / results
        write_log_metrics(model_name, epoch, tr_loss, tr_acc, tr_auc,
                            val_loss, val_acc, val_auc)

        # TEST PREDICTIONS
        # this writes our ACTUAL test predictions from the best 
        # NOTE; we are now testing every epoch but these are the real ones 
        if val_auc > best_val_auc:
            write_test_predictions(model_name, te_probs, test_dataset)
            

        # PATIENCE
        if check_patience(train_losses):
            print("PATIENCE REACHED")
            break
