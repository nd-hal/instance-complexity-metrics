import argparse
import itertools


###################################################################
################### Parsing Arguments from CLI ####################
###################################################################
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
# e.g., --fclass='bert'
def get_inputs_from_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fclass", type=str)
    parser.add_argument("--score", type=str)
    parser.add_argument("--rev", type=str2bool, default=False)
    parser.add_argument("--off", type=int, default=0)
    parser.add_argument("--ftune", type=str2bool, default=True)
    parser.add_argument("--sub", type=str2bool, default=False)

    args = parser.parse_args()
    function_class, score_var, rev, offset, finetune, sub = args.fclass, args.score, args.rev, args.off, args.ftune, args.sub

    print('ARGS ARE:')
    print(f"function_class: {function_class}, score_var: {score_var}, rev: {rev}, offset: {offset}, finetune: {finetune}, sub: {sub}")

    return function_class, score_var, rev, offset, finetune, sub


###################################################################
################### Setting the Hyperparameters ###################
###################################################################
# NOTE; we will edit this to change the learning rates, etc.
def setup_params(function_class, score_var, rev=False, offset=0, finetune=False):
    # list of currently supported transformer models
    if function_class in ['bert', 'roberta', 'llama', 'mistral']:
        # embeddings
        embedding_d = {
            'maxLength': [256],
            'embType': ['HF']
        }

        if function_class == 'bert':
            this_checkpoint = 'bert-base-uncased'
            model_param_d = {
                'learningRate': [3e-5, 3e-6]
            }

        elif function_class == 'roberta':
            this_checkpoint = 'tner/xlm-roberta-base-uncased-all-english'
            model_param_d = {
                'learningRate': [3e-5, 3e-6]
            }

        elif function_class == 'llama':
            # this_checkpoint = 'TheBloke/Llama-2-7B-GPTQ'
            this_checkpoint = 'TheBloke/toxicqa-Llama2-7B-GPTQ'
            # NOTE; lower priors on llama learning rate
            model_param_d = {
                # 'r': [16, 32, 64],
                # 'loraAlpha': [8, 16, 32],
                # 'loraDropout': [0.05],
                'learningRate': [4e-4, 4e-5]
            }

        elif function_class == 'mistral':
            this_checkpoint = 'TheBloke/Mistral-7B-v0.1-GPTQ'
            model_param_d = {
                # 'r': [16, 32, 64],
                # 'loraAlpha': [8, 16, 32],
                # 'loraDropout': [0.05],
                'learningRate': [4e-4, 4e-5]
            }

        else:
            print("ERROR: current supported models are ['ffn', 'cnn', 'lstm', 'bert', 'roberta', 'llama']")


    # non transformer models -- no HF checkpoints / entirely local
    else:
        # use word2vec embeddings
        embedding_d = {
            'maxLength': [256],
            'embType': ['w2v']      # note defaulting to word2vec
        }

        if function_class == 'ffn':
            this_checkpoint = 'local-ffn'
            # ffn (6)
            model_param_d = {
                'numLayers': [1, 3, 5],
                'layerNodes': [64, 128],
                'learningRate': [1e-3]
            }

        elif function_class == 'cnn':
            this_checkpoint = 'local-cnn'
            # cnn (6)
            model_param_d = {
                'numLayers': [1, 3, 5],
                'numFilters': [16, 32],
                'kernelSize': [5],
                'learningRate': [1e-3]
            }

        elif function_class == 'lstm':
            this_checkpoint = 'local-lstm'
            # lstm (6)
            model_param_d = {
                'numLstmLayers': [1, 2, 3],
                'lstmNodes': [32, 64],
                'learningRate': [1e-3],
            }

        else:
            print("ERROR: current supported models are ['ffn', 'cnn', 'lstm', 'bert', 'roberta', 'llama']")


    # smaller batch size on LLMs since CRC a10's anbd a40's can't fit larger than 4(?)
    this_batch_size = 8 if function_class in ['llama', 'mistral'] else 16
    # 1 epoch allows us to only do inference
    these_epochs = 15 if finetune else 1


    # training parameters
    train_param_dlst = {
        'modelCheckpoint': [this_checkpoint],
        # NOTE; these are CL options
        'difficultTime': ['None', 'Constant'],
        # 'difficultTime': ['None', 'Constant', 'First', 'Second'],
        'epochs': [these_epochs],
        'batchSize': [this_batch_size],
        'scoreVariable': [score_var]
    }

    # put all parameters together into one large dict to pass as arguments
    all_param_d = dict(embedding_d, **model_param_d, **train_param_dlst)
    # which keys actually vary during the grid search
    grid_keys = { k:v for k,v in all_param_d.items() if len(v)>1 }
    # create a list of all combinations to train
    all_param_lst = list( itertools.product(*all_param_d.values())  )


    return {
        'rev': rev,
        'offset': offset,
        'all_param_d': all_param_d,
        'grid_keys': grid_keys,
        'all_param_lst': all_param_lst,
        'model_param_d': model_param_d
    }

# set the name of the model based on the hyperparameters
def setup_model_name(this_offset, these_params, grid_keys, modelCheckpoint, scoreVariable, finetune=False, subset=False):
    sub_params = { k:v for k,v in these_params.items() if k in grid_keys.keys() }
    sub_modelCheckpoint = modelCheckpoint.split('/')[-1]
    model_name = sub_modelCheckpoint + '_' + scoreVariable + '_ft-' + str(finetune) + '_sub-' + str(subset) + '_t' + str(this_offset) + '_' + '_'.join([f"{k}-"+str(e) for k,e in sub_params.items()])

    return model_name


###################################################################
################### Writing Outputs ###############################
###################################################################
# writes the metrics for the log files -- use these later for performance/results
def write_log_metrics(model_name, epoch, train_loss, train_acc, train_auc,
                        val_loss, val_acc, val_auc):
    with open(f"logs/{model_name}_log.txt", "a") as outputFile:
        outputFile.write(f"{epoch}\t{train_loss:.4f}\t{train_acc:.4f}\t{train_auc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{val_auc:.4f}\n")

# writes the predictions for the test set -- includes Text due to shuffling issue
def write_test_predictions(model_name, test_probs, test_dataset):
    tdf = test_dataset.data
    with open("predictions/test/" + str(model_name) + "_predictions.txt", "w") as predictionsFile:
        predictionsFile.write("Id|Text|Prediction\n")
        for i in range(0, len(test_probs)):
            # ensure we have the text too
            predictionsFile.write(str(i) + "|" + str(tdf.loc[i, 'text']) + "|" + str(test_probs[i]) + "\n")