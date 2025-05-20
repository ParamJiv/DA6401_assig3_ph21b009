import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler

import os
import yaml
from tqdm.notebook import tqdm
import wandb
from typing import Dict, List, Tuple
import gc

import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# This function is used to get the processed outout without padding for calculating loss and back-prop
def get_truncated_outputs(model_output, true_output, tensor_len):

    # for storing all the outputs
    real_outputs = []
    real_truth = []

    # for getting the truncated output
    for i in range(len(tensor_len)):
        real_outputs.append( model_output[i,1:tensor_len[i], :] )
        real_truth.append( true_output[i,1:tensor_len[i]] )

    return real_outputs, real_truth # List[ Tensor[batch_size, trg_len - 1, vocab_size] ], List[ Tensor[batch_size, trg_len - 1] ]

def get_correct_output_count(model_truncated_outputs : List, true_truncated_outputs):

    batch_size = len(model_truncated_outputs)
    correct_outputs = 0
    total_chars = 0
    correct_words = 0

    # for getting the correct output
    for i in range(batch_size):
        current_correct = (model_truncated_outputs[i].argmax(-1).cpu() == true_truncated_outputs[i].cpu()).sum().item()
        correct_outputs += (current_correct)
        total_chars += model_truncated_outputs[i].shape[0]

        correct_words += (model_truncated_outputs[i].shape[0] == current_correct)
    
    # print(correct_words, batch_size)

    return correct_outputs, total_chars, correct_words

def compute_seq_loss(outputs, targets, pad_idx, sos_idx=None):
    """
    outputs: [batch_size, trg_len - 1, vocab_size]  (predictions from decoder)
    targets: [batch_size, trg_len]  (original decoder inputs with <sos> at index 0)

    This function aligns outputs to targets[:, 1:], and ignores <pad> and optionally <eos>.
    """
    # Align outputs with target tokens after <sos>
    outputs = outputs.reshape(-1, outputs.size(-1))          # [B*(T-1), V]
    targets = targets[:, :].reshape(-1)                     # [B*(T-1)]

    if sos_idx is not None:
        # Mask out both <pad> and <sos>
        mask = (targets != pad_idx) & (targets != sos_idx)
        # print(mask.shape, outputs.shape, targets.shape)
        outputs = outputs[mask]
        targets = targets[mask]
        loss_fn = nn.CrossEntropyLoss()
    else:
        # Just ignore <pad>
        loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

    return loss_fn(outputs, targets)



class Trainer():
    def __init__(self, model,  
                train_dataloader,
                val_dataloader,
                optimizer_params : Dict = {"lr" : 1e-4}, ):
        
        # Initializing attributes
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer_params = optimizer_params

        # defining the loss fn
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = 0)

        # defining the optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), **self.optimizer_params)
        # defining the learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',        
            factor=0.5,            
            patience=2,            # after 2 epochs without improvement
            verbose=True,
            min_lr=1e-6  
        )

        print(f"Total words in train dataset : {len(self.train_dataloader)}")
        print(f"Total words in val dataset : {len(self.val_dataloader)}")
        print(f"Total parameters in the model :  {sum([p.numel() for p in self.model.parameters()])*1e-3:.3f} K")

        print(f"Total parameters in encoder : {sum([p.numel() for p in self.model.encoder.parameters()])*1e-3:.3f} K")
        print(f"Total parameters in decoder : {sum([p.numel() for p in self.model.decoder.parameters()])*1e-3:.3f} K")

    def compute_seq_loss(self, outputs, targets, pad_idx, sos_idx=None):
        """
        outputs: [batch_size, trg_len - 1, vocab_size]  (predictions from decoder)
        targets: [batch_size, trg_len]  (original decoder inputs with <sos> at index 0)

        This function aligns outputs to targets[:, 1:], and ignores <pad> and optionally <eos>.
        """
        # Align outputs with target tokens after <sos>
        outputs = outputs.reshape(-1, outputs.size(-1))          # [B*(T-1), V]
        targets = targets[:, :].reshape(-1)                     # [B*(T-1)]

        if sos_idx is not None:
            # Mask out both <pad> and <sos>
            mask = (targets != pad_idx) & (targets != sos_idx)
            # print(mask.shape, outputs.shape, targets.shape)
            outputs = outputs[mask]
            targets = targets[mask]
            loss_fn = nn.CrossEntropyLoss()
        else:
            # Just ignore <pad>
            loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

        return loss_fn(outputs, targets)
    
    def validate_model(self):

        self.model.eval()
        val_loss = 0
        val_accuracy = 0
        val_word_matches = 0
        total_items = 0

        with torch.no_grad():

            for (native_tensor, native_input_length, roman_tensor, roman_input_length) in self.val_dataloader:

                # moving the native tensor to the device
                native_tensor = native_tensor.to(device)
                roman_tensor = roman_tensor.to(device)

                outputs = self.model(native_tensor, native_input_length, roman_tensor)
                actual_model_output, actual_truth = get_truncated_outputs(outputs, roman_tensor, roman_input_length)

                # getting the count of correct answers
                temp_correct, temp_count, temp_correct_words = get_correct_output_count(actual_model_output, actual_truth)
                val_accuracy += temp_correct ; total_items += temp_count
                val_word_matches += (temp_correct_words)

                # calculating the loss
                # loss = self.loss_fn(torch.concatenate(actual_model_output), torch.concatenate(actual_truth))
                loss = self.compute_seq_loss(outputs, roman_tensor, pad_idx = 0, sos_idx = 1)
                val_loss += loss.item()

                # memory management
                del native_tensor, native_input_length, roman_tensor, roman_input_length, loss
                torch.cuda.empty_cache()
                gc.collect()
        
        return val_loss/len(self.val_dataloader) , (val_accuracy/total_items) , (val_word_matches)

    def train_model(self, epochs : int = 10):

        
        train_loss_history = []
        train_accuracy_history = []
        train_word_matches_history = []

        val_loss_history = []
        val_accuracy_history = []
        val_word_matches_history = []

        loop_obj = (range(epochs))

        for epoch in loop_obj:

            train_loss = 0
            train_accuracy = 0
            total_items = 0
            train_word_matches = 0
            self.model.train()

            loop_obj_dataloader = tqdm(self.train_dataloader)

            for (native_tensor, native_input_length, roman_tensor, roman_input_length) in loop_obj_dataloader:

                # moving the native tensor to the device
                native_tensor = native_tensor.to(device)
                roman_tensor = roman_tensor.to(device)

                outputs = self.model(native_tensor, native_input_length, roman_tensor)
                actual_model_output, actual_truth = get_truncated_outputs(outputs, roman_tensor, roman_input_length)

                # getting the count of correct answers
                temp_correct, temp_count, temp_correct_words = get_correct_output_count(actual_model_output, actual_truth)
                train_accuracy += temp_correct ; total_items += temp_count
                train_word_matches += (temp_correct_words)
                # total_items += len(native_tensor)

                # calculating the loss
                # loss = self.loss_fn(torch.concatenate(actual_model_output), torch.concatenate(actual_truth))
                loss = self.compute_seq_loss(outputs, roman_tensor, pad_idx = 0, sos_idx = 1)
                train_loss += loss.item()
                
                # optimizer 
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # memory management
                del native_tensor, native_input_length, roman_tensor, roman_input_length, loss
                loop_obj_dataloader.set_description(f"correct in train : {train_accuracy}")
                torch.cuda.empty_cache()
                gc.collect()

            # adding the train and val : loss and accuracy
            train_loss_history.append(train_loss/len(self.train_dataloader))
            train_accuracy_history.append(train_accuracy/total_items)
            train_word_matches_history.append(train_word_matches/len(self.train_dataloader.dataset))

            # doing validation
            val_loss, val_accuracy, val_word_matches = self.validate_model()
            val_loss_history.append(val_loss)  
            val_accuracy_history.append(val_accuracy)
            val_word_matches_history.append(val_word_matches/len(self.val_dataloader.dataset))

            # loop_obj.set_description(f"Epoch {epoch+1}/{epochs}")
            # loop_obj.set_postfix(
            #     Train_loss = train_loss_history[-1],
            #     Train_accuracy = train_accuracy_history[-1],
            #     Val_loss = val_loss_history[-1],
            #     Val_accuracy = val_accuracy_history[-1],
            # )

            print(f"""Epoch {epoch+1}/{epochs} : Train_loss = {train_loss_history[-1]}, Train_accuracy = {train_accuracy_history[-1]}, Val_loss = {val_loss_history[-1]}, Val_accuracy = {val_accuracy_history[-1]}
            Train word matches : {train_word_matches}, Val word matches : {val_word_matches}""")

            # step the scheduler, after calculating val loss
            self.scheduler.step(val_loss)

        self.train_loss_history = train_loss_history
        self.train_accuracy_history = train_accuracy_history
        self.val_loss_history = val_loss_history
        self.val_accuracy_history = val_accuracy_history

        return {"train_loss" : train_loss_history,
                "train_accuracy" : train_accuracy_history, 
                "val_loss" : val_loss_history, 
                "val_accuracy" : val_accuracy_history,
                "train_word_matches" : train_word_matches_history,
                "val_word_matches" : val_word_matches_history}