import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
import yaml

# Import your modules (adjust these to your package structure)
from data_loader import Dakshina_Dataset
from data_loader import get_loaders
from train import build_vocab, load_data
from models.vanilla import Encoder, Decoder, Seq_2_Seq
from models.attention import Encoder_for_attention, Decoder_with_attention, Seq_2_Seq_with_attention


def load_params(params_path):
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    return params

def load_vocab(vocab_path):
    # Assuming vocab is saved as a pickle or json
    import pickle
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--s2s_weights', type=str, required=True, help='Path to encoder weights')
    parser.add_argument('--params_path', type=str, required=True, help='Path to params.yaml')
    parser.add_argument('--native_vocab_path', type=str, required=True, help='Path to native vocab pickle')
    parser.add_argument('--roman_vocab_path', type=str, required=True, help='Path to roman vocab pickle')
    parser.add_argument('--roman_idx2char_path', type=str, required=True, help='Path to roman idx2char pickle')
    parser.add_argument('--use_attention', action='store_true', help='Use attention model')
    parser.add_argument('--output_path', type=str, default='predictions.csv', help='Path to save predictions CSV')
    args = parser.parse_args()

    # Load params and vocabs
    params = load_params(args.params_path)
    native_char2ind = load_vocab(args.native_vocab_path)
    roman_char2ind = load_vocab(args.roman_vocab_path)
    roman_idx2char = load_vocab(args.roman_idx2char_path)

    device = params.get('device', 'cpu')

    # Load test data
    # Load data and vocabs (implement this function as per your notebook)
    train_df, val_df, test_df = load_data(
        args.train_path, args.val_path, args.test_path
    )

    native_char2ind, native_idx2char = build_vocab(train_df["native"].tolist())
    roman_char2ind, roman_idx2char = build_vocab(train_df["roman"].tolist())

    # Get dataloaders
    train_loader, val_loader, test_dataloader = get_loaders(
        train_df, val_df, test_df, native_char2ind, roman_char2ind, 
        batch_size=params['batch_size']
    )

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
    model.eval()

    # Prediction loop
    test_predictions = []
    with torch.no_grad():
        for test_native_tensor, test_native_tensor_len, test_roman_tensor, test_roman_tensor_len in test_dataloader:
            test_native_tensor = test_native_tensor.to(device)
            test_native_tensor_len = test_native_tensor_len.to(device)
            test_roman_tensor = test_roman_tensor.to(device)
            outputs = model(test_native_tensor, test_native_tensor_len, test_roman_tensor)
            for word in outputs.argmax(-1):
                output_word = ""
                word = word[1:]  # skip <sos>
                for i, v in enumerate((word != 0) & (word != 1) & (word != 2)):  # skip pad, sos, eos
                    if not v:
                        break
                    output_word += roman_idx2char[word[i].item()]
                test_predictions.append(output_word)

    # Save predictions
    pred_df = test_df.copy()
    pred_df['pred'] = test_predictions
    pred_df.to_csv(args.output_path, index=False)
    print(f"Predictions saved to {args.output_path}")

if __name__ == "__main__":
    main()
