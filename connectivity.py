import torch.nn.functional as F
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import yaml
import os

# Import your modules (adjust these to your package structure)
from data_loader import Dakshina_Dataset
from models.vanilla import Encoder, Decoder, Seq_2_Seq
from models.attention import Encoder_for_attention, Decoder_with_attention, Seq_2_Seq_with_attention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_lstm_connectivity(input_word: str, s2s_lstm: torch.nn.Module, tamil_vocab: Dict[str, int], 
                               roman_vocab: Dict[str, int], max_output_len: int = 10,) :
    """
    Visualizes the connectivity |∂h_{o,t'}/∂x_t|^2 of a Seq2Seq LSTM model, showing which input Tamil character
    influences each predicted Roman character, as defined in the Distill article.
    
    """
    # Inverse vocabularies for decoding
    print(f"Input tamil word : {input_word}")
    inv_tamil_vocab = {v: k for k, v in tamil_vocab.items()}
    inv_roman_vocab = {v: k for k, v in roman_vocab.items()}
    
    # Convert input word to indices and tensor
    input_indices = [tamil_vocab.get(c, tamil_vocab.get('<unk>', 0)) for c in input_word]
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)  # Shape: (1, seq_len)
    input_lengths = torch.tensor([len(input_indices)]).to(device)
    
    # Ensure model is in evaluation mode but enable gradients for Jacobian computation
    s2s_lstm.train()
    
    # Get input embeddings (requires access to encoder's embedding layer)
    embedding_layer = s2s_lstm.encoder.embedding if hasattr(s2s_lstm.encoder, 'embedding') else None
    if embedding_layer is None:
        raise AttributeError("Encoder must have an 'embedding' layer for input embeddings.")
    
    # Enable gradients for input embeddings
    input_tensor
    input_embeds = embedding_layer(input_tensor).requires_grad_(True)  # Shape: (1, seq_len, embed_size)
    input_embeds.retain_grad()
    s2s_lstm.encoder.embedded.requires_grad_(True)
    s2s_lstm.encoder.embedded.retain_grad()
    
    # Forward pass through encoder
    encoder_outputs, (hidden, cell) = s2s_lstm.encoder(input_tensor, input_lengths)
    # encoder_outputs: (1, seq_len, hidden_size), hidden/cell: (num_layers, 1, hidden_size)
    
    # Initialize decoder input (start token) and states
    decoder_input = torch.tensor([roman_vocab.get('<sos>', 1)], dtype=torch.long).to(device)  # Shape: (1, 1)
    hidden = hidden.to(device)
    cell = cell.to(device)
    
    predicted_chars = []
    connectivity_weights = []
    
    for t_prime in range(max_output_len):
        # Forward pass through decoder
        s2s_lstm.encoder.embedded.retain_grad()
        output, (hidden, cell) = s2s_lstm.decoder(decoder_input, (hidden, cell))
        # output: (1, 1, output_size)
        
        # Predict character
        pred_char_idx = output.squeeze(1).argmax(dim=-1).item()
        predicted_char = inv_roman_vocab.get(pred_char_idx, '<unk>')
        predicted_chars.append(predicted_char)
        
        # Compute connectivity: |∂h_{o,t'}/∂x_t|^2
        # h_{o,t'} is the top-layer decoder hidden state
        decoder_hidden = hidden[-1].squeeze(0)  # Shape: (1, hidden_size)
        
        # Zero out previous gradients
        if s2s_lstm.encoder.embedded.grad is not None:
            # input_embeds.grad.zero_()
            s2s_lstm.encoder.embedded.grad.zero_()
        
        # Compute gradients of decoder_hidden w.r.t. input_embeds
        decoder_hidden.sum().backward(retain_graph=True)
        
        # Get gradients ∂h_{o,t'}/∂x_t
        gradients = s2s_lstm.encoder.embedded.grad.squeeze(0)  # Shape: (seq_len, embed_size)
        
        # Compute squared norm of gradients for each input position t
        connectivity = torch.norm(gradients, dim=-1) ** 2  # Shape: (seq_len,)
        connectivity = connectivity.cpu().numpy()
        
        # Normalize connectivity weights to sum to 1 for visualization
        connectivity = connectivity / (connectivity.sum() + 1e-8)
        connectivity_weights.append(connectivity)
        
        # Prepare next decoder input
        decoder_input = output.argmax(dim=-1)
        
        # Stop if end token is predicted
        if pred_char_idx == roman_vocab['<pad>'] or pred_char_idx == roman_vocab["<eos>"]:
            break
    
    # Stack connectivity weights
    connectivity_weights = np.stack(connectivity_weights)  # Shape: (output_len, input_len)
    
    # Create heatmap visualization
    plt.figure(figsize=(max(8, len(input_word) * 0.5), max(6, len(predicted_chars) * 0.5)))
    sns.heatmap(
        connectivity_weights,
        xticklabels=[c for c in input_word],
        yticklabels=predicted_chars,
        cmap='viridis',
        annot=True,
        fmt='.2f',
        cbar_kws={'label': 'Connectivity |∂h_{o,t\'}/∂x_t|^2'},
        square=False
    )
    plt.xlabel('Input Tamil Characters')
    plt.ylabel('Predicted Roman Characters')
    plt.title(f'Connectivity for "{input_word}"')
    plt.tight_layout()
    
    # Save heatmap
    heatmap_path = os.path.dirname(__file__) + f'/connectivity images/lstm_connectivity_heatmap_{input_word}.png'
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')

    plt.show()
    plt.close()
    
    return predicted_chars, heatmap_path

def load_params(params_path):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_vocab(vocab_path):
    import pickle
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

root = os.path.dirname(__file__)
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2s_weights', type=str, required=True, help='Path to encoder weights')
    parser.add_argument('--params_path', type=str, required=True, help='Path to params.yaml')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model')
    parser.add_argument('--native_vocab_path', type=str, required=True, help='Path to native vocab pickle')
    parser.add_argument('--roman_vocab_path', type=str, required=True, help='Path to roman vocab pickle')
    parser.add_argument('--roman_idx2char_path', type=str, required=True, help='Path to roman idx2char pickle')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions CSV')
    args = parser.parse_args()

    # Load params and vocabs
    params = load_params(args.params_path)
    native_char2ind = load_vocab(args.native_vocab_path)
    roman_char2ind = load_vocab(args.roman_vocab_path)
    roman_idx2char = load_vocab(args.roman_idx2char_path)

    device = params.get('device', 'cpu')

    # Load model
    if args.use_attention:
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
            pad_idx=roman_char2ind['<pad>']
        )
        model = Seq_2_Seq_with_attention(encoder, decoder, device=device)
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
        model = Seq_2_Seq(encoder, decoder, device=device)

    model = model.to(device).load_state_dict(torch.load(args.s2s_weights, map_location=device))
    model.train()

    visualize_lstm_connectivity(input_word="அகராதி",
                                s2s_lstm = model,
                                tamil_vocab=native_char2ind,
                                roman_vocab=roman_char2ind,)


if __name__ == "__main__":
    main()