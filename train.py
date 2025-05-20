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

import argparse
import os
import yaml
import wandb

# Import modules 
from data_loader import get_loaders
from models.vanilla import Encoder, Decoder, Seq_2_Seq
from models.attention import Encoder_for_attention, Decoder_with_attention, Seq_2_Seq_with_attention, Attention_module
from trainer import Trainer 

def build_vocab(word_list, add_special_tokens=True):
    chars = sorted(set(char for word in word_list for char in word))
    # print(chars)
    
    if add_special_tokens:
        vocab = ['<pad>', '<sos>', '<eos>', '<unk>'] + chars
    else:
        vocab = ['<unk>'] + chars
    
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return char2idx, idx2char

def load_data(train_data_path, test_data_path, val_data_path ):

    ## Loading the train, test, val data
    train_data_list = []
    with open(train_data_path, encoding='utf-8') as f:
            for line in f:
                native, roman, freq = line.strip().split('\t')
                train_data_list.append((native, roman, freq))

    print(f"Number of training examples: {len(train_data_list)}")

    test_data_list = []
    with open(test_data_path, encoding="utf-8") as f:
        for line in f:
                native, roman, freq = line.strip().split('\t')
                test_data_list.append((native, roman, freq))

    print(f"Number of test examples: {len(test_data_list)}")

    val_data_list = []
    with open(val_data_path, encoding="utf-8") as f:
        for line in f:
                native, roman, freq = line.strip().split('\t')
                val_data_list.append((native, roman, freq))

    print(f"Number of val examples: {len(val_data_list)}")

    train_df = pd.DataFrame(train_data_list, columns=["native", "roman", "freq"])
    test_df = pd.DataFrame(test_data_list, columns=["native", "roman", "freq"])
    val_df = pd.DataFrame(val_data_list, columns=["native", "roman", "freq"])

    return train_df, test_df, val_df

root = os.path.dirname(__file__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default=root + '\dataset\dakshina_dataset_v1.0\\ta\lexicons\\ta.translit.sampled.train.tsv', help='Path to train CSV')
    parser.add_argument('--val_path', type=str, default=root + '\dataset\dakshina_dataset_v1.0\\ta\lexicons\\ta.translit.sampled.dev.tsv', help='Path to val CSV')
    parser.add_argument('--test_path', type=str, default=root + '\dataset\dakshina_dataset_v1.0\\ta\lexicons\\ta.translit.sampled.test.tsv', help='Path to test CSV')
    parser.add_argument('--params_path', type=str, default=root + '\params.yaml', help='Path to params.yaml')
    parser.add_argument('--wandb_api_key', type=str, required=True, help='Wandb API key')
    parser.add_argument('--wandb_project', type=str, default='transliteration', help='Wandb project name')
    parser.add_argument('--use_attention', default=False, action='store_true', help='Use attention model')
    args = parser.parse_args()

    # Wandb login
    wandb.login(key=args.wandb_api_key)

    # Load params
    with open(args.params_path, 'r') as f:
        params = yaml.safe_load(f)

    # Load data and vocabs (implement this function as per your notebook)
    train_df, val_df, test_df = load_data(
        args.train_path, args.val_path, args.test_path
    )

    native_char2ind, native_idx2char = build_vocab(train_df["native"].tolist())
    roman_char2ind, roman_idx2char = build_vocab(train_df["roman"].tolist())

    # Get dataloaders
    train_loader, val_loader, test_loader = get_loaders(
        train_df, val_df, test_df, native_char2ind, roman_char2ind, 
        batch_size=params['batch_size']
    )

    # Model selection
    if args.use_attention:
        att = Attention_module(params['encoder_layers'], params['decoder_layers'])
        encoder = Encoder_for_attention(
            input_vocab_size=len(native_char2ind),
            embedding_size=params['embedding_size'],
            hidden_size=params['hidden_size'],
            encoder_layers=params['encoder_layers'],
            rnn_type=params['cell_type'],
            dropout=params['dropout'],
            pad_idx=native_char2ind['<pad>']
        )
        decoder = Decoder_with_attention(
            output_vocab_size=len(roman_char2ind),
            embedding_size=params['embedding_size'],
            hidden_size=params['hidden_size'],
            decoder_layers=params['decoder_layers'],
            rnn_type=params['cell_type'],
            dropout=params['dropout'],
            pad_idx=roman_char2ind['<pad>'],
            attention = att,
        )
        model = Seq_2_Seq_with_attention(encoder, decoder)
    else:
        encoder = Encoder(
            input_vocab_size=len(native_char2ind),
            embedding_size=params['embedding_size'],
            hidden_size=params['hidden_size'],
            encoder_layers=params['encoder_layers'],
            rnn_type=params['cell_type'],
            dropout=params['dropout'],
            pad_idx=native_char2ind['<pad>']
        )
        decoder = Decoder(
            output_vocab_size=len(roman_char2ind),
            embedding_size=params['embedding_size'],
            hidden_size=params['hidden_size'],
            decoder_layers=params['decoder_layers'],
            rnn_type=params['cell_type'],
            dropout=params['dropout'],
            pad_idx=roman_char2ind['<pad>']
        )
        model = Seq_2_Seq(encoder, decoder)

    # Start wandb run
    wandb.init(project=args.wandb_project, config=params)

    # Train the model (implement this function/class as per your notebook)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    history = trainer.train_model(params['epochs'])

    for i in range(len(history["train_loss"])):
        wandb.log({
            "train_loss": history["train_loss"][i],
            "train_accuracy": history["train_accuracy"][i],
            "val_loss": history["val_loss"][i],
            "val_accuracy": history["val_accuracy"][i],
            "train_word_matches": history["train_word_matches"][i],
            "val_word_matches": history["val_word_matches"][i],
        })
    
    wandb.finish()

if __name__ == '__main__':
    main()
