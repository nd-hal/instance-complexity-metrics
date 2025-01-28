import os
import copy
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import *

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

from bitsandbytes.optim import AdamW

import itertools
import pickle
from copy import deepcopy
import subprocess
import sys

import keras

# NOTE; removed QLoRA due to versioning issues and similarity to BERT models
# import peft
# from peft import get_peft_model, LoraConfig, TaskType
# import accelerate
# # import auto_gptq
# from transformers.utils import is_auto_gptq_available,   is_optimum_available
# print(is_auto_gptq_available())
# print(is_optimum_available())


###################################################################
################### Setting up Model and Tokenizer ################
###################################################################
# main function to create any transformer from model checkpoint
def setup_transformer(train_dataset, train_dataloader, modelCheckpoint, epochs, access_token,
                      learningRate, r=None, loraAlpha=None, loraDropout=None):
    # HF models need a few more args for from_pretrained()
    if 'bert' not in modelCheckpoint:
        add_setup_d = {
            'token': access_token,
            'device_map': "cuda",
            'offload_folder': "offload",
            'trust_remote_code': True
        }
    else:
        add_setup_d = {}

    model = AutoModelForSequenceClassification.from_pretrained(modelCheckpoint, num_labels=len(set(train_dataset.labels)),
                                                               torch_dtype=torch.bfloat16, **add_setup_d)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # check local on mac
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")

    # test_input_ids, test_attention_mask = get_data_prepare_for_eval(testText, tokenizer, max_length, device)
    model.to(device)

    # need to add an EOS token to avoid batch inference issue
    if 'bert' not in modelCheckpoint:
        model.config.pad_token_id = model.config.eos_token_id
        # optimizer = AdamW(model.parameters(), lr=learningRate,
        #               percentile_clipping=40, weight_decay=0.04)

        optimizer = AdamW(model.parameters(), lr=learningRate)

        # NOTE; can be enabled to apply QLoRA
        # # apply LoRA
        # peft_config = LoraConfig(
        #     task_type=TaskType.SEQ_CLS, r=r, lora_alpha=loraAlpha, lora_dropout=loraDropout, bias="none",
        #     target_modules=[
        #         "q_proj",
        #         "v_proj",
        #     ],
        # )

        # model = get_peft_model(model, peft_config)
        
    else:
        optimizer = AdamW(model.parameters(), lr=learningRate)
    
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    return model, device, optimizer, total_steps, scheduler


# similarly set up the tokenizer
def setup_tokenizer(model_checkpoint, access_token):
    pretrained_models_dir = './pretrained_models_dir'
    if not os.path.isdir(pretrained_models_dir):
        os.mkdir(pretrained_models_dir)   # directory to save pretrained models

    cache_dir = os.path.join(pretrained_models_dir, model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir=cache_dir, token=access_token)
    if 'bert' not in model_checkpoint:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


###################################################################
################### Run a Training Epoch ##########################
###################################################################
# main function to create any transformer from model checkpoint
def train_transformer(model, device, optimizer, scheduler, train_dataloader,
               epoch, epochs):
    print('TRAINING')
    tr_preds, tr_probs, tr_true, tr_losses, tr_IDs, tr_texts = [], [], [], [], [], []
    bdx = 0
    total_train_loss = 0

    model.train()
    for batch in train_dataloader:
        bdx += 1
        # if bdx % round(len(train_dataloader)*.1) == 0:
        if bdx % 10 == 0:
            print(f'Batch: {bdx} / {len(train_dataloader)}')
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        ID = batch['ID']
        text = batch['text']

        with torch.autocast(device_type=str(device), dtype=torch.bfloat16): 
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss    # store the loss
        loss.backward()

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).detach().cpu().float().numpy()
        probs = torch.softmax(logits, dim=1).detach().cpu().float().numpy()
        tr_preds.extend(preds)    # store the preds / probs
        tr_probs.extend(probs)

        labels_cpu = labels.detach().cpu().numpy()
        tr_true.extend(labels_cpu)
        these_tr_losses = keras.losses.sparse_categorical_crossentropy(labels_cpu, probs).numpy()
        tr_losses.extend(these_tr_losses)
        tr_IDs.extend(ID)
        tr_texts.extend(text)

        # print(f'preds: {preds}')
        # print(f'probs: {probs}')

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    # Compute AUC and accuracy
    tr_true_onehot = np.eye(model.config.num_labels)[tr_true]
    tr_probs = np.array(tr_probs)

    auc = roc_auc_score(tr_true_onehot, tr_probs, multi_class='ovr')
    accuracy = accuracy_score(tr_true, tr_preds)
    loss = total_train_loss / len(train_dataloader)     # calculate avg train loss

    print(f"Epoch {epoch}/{epochs}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    print("-----")

    return float(loss), accuracy, auc, list(tr_probs[:,1]), tr_preds, tr_true, tr_losses, tr_IDs, tr_texts

# custom binary cross-entropy loss using torch
def calculate_val_loss(inputs, labels):
    m = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCELoss()
    t_input = torch.tensor(inputs, dtype=torch.float)
    t_labels = torch.tensor(labels, dtype=torch.float)

    return loss_fn(m(t_input), t_labels)


###################################################################
################### Run Inference #################################
###################################################################
def validate_transformer(model, device, this_dataloader, epoch, epochs):
    print('INFERENCE')
    # Validation
    model.eval()
    val_preds, val_probs, val_true, val_losses, val_IDs, val_texts = [], [], [], [], [], []
    total_val_loss = 0
    for bdx, batch in enumerate( this_dataloader ):
        print(f" {bdx} / {len(this_dataloader)} ")
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        ID = batch['ID']
        text = batch['text']

        with torch.autocast(device_type=str(device), dtype=torch.bfloat16):
            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        local_logits = logits.detach().cpu().float().numpy()
        # replace nan in local_logits -- huggingface bug
        if any(np.isnan(local_logits).reshape(-1,1)):
            with open(f'WARNINGS_{str(model).split("(")[0]}.txt', 'a') as f:
                f.write(f"WARNING: model returned np.nan in epoch {epoch}\n")
            local_logits = np.nan_to_num(local_logits, copy=True, nan=0)
        # NOTE; bfloat16 requires a cast back to .float() before .numpy()
        preds = torch.argmax(torch.from_numpy(local_logits), dim=1).float().numpy()
        probs = torch.softmax(torch.from_numpy(local_logits), dim=1).float().numpy()
        # preds = torch.argmax(logits, dim=1).detach().cpu().float().numpy()
        # probs = torch.softmax(logits, dim=1).detach().cpu().float().numpy()
        labels_cpu = labels.detach().cpu().numpy()

        val_preds.extend(preds)
        val_probs.extend(probs)
        val_true.extend(labels_cpu)
        val_IDs.extend(ID)
        val_texts.extend(text)

        # if np.mean(val_true) != 0:
        loss = calculate_val_loss(probs[:, 1], labels_cpu)
        total_val_loss += loss
        these_val_losses = keras.losses.sparse_categorical_crossentropy(labels_cpu, probs).numpy()
        val_losses.extend(these_val_losses)

    # Compute AUC and accuracy
    val_probs = np.array(val_probs)
    val_true_onehot = np.eye(model.config.num_labels)[val_true]
    if np.mean(val_true) != 0:
        auc = roc_auc_score(val_true_onehot, val_probs, multi_class='ovr')
        accuracy = accuracy_score(val_true, val_preds)
        loss = total_val_loss / len(this_dataloader)
    else:
        auc, accuracy, loss = np.nan, np.nan, np.nan

    print(f"Epoch {epoch}/{epochs}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Loss: {loss:.4f}")
    print("-----")

    return float(loss), accuracy, auc, list(val_probs[:,1]), val_preds, val_true, val_losses, val_IDs, val_texts