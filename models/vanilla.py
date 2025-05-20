import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_vocab_size, 
                 embedding_size : int ,
                 hidden_size : int, 
                 encoder_layers : int = 1,
                 rnn_type = "LSTM",
                 dropout = 0.2,
                 pad_idx = 0):
        
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encoder_layers = encoder_layers
        self.rnn_type = rnn_type
        self.pad_idx = pad_idx
        self.dropout = dropout

        self.embedding = nn.Embedding(input_vocab_size, embedding_size, padding_idx=pad_idx)

        # initialize the RNN
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, encoder_layers, batch_first=True, dropout = dropout)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, encoder_layers, batch_first=True, dropout = dropout)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, encoder_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")

    def forward(self, input_idxs, input_lengths):
        # x: [batch, seq_len]
        self.embedded = nn.Dropout(self.dropout)(self.embedding(input_idxs))  # shape: [batch, seq_len, emb_dim]
        self.packed = nn.utils.rnn.pack_padded_sequence(self.embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False) # for avoiding paddings
        self.outputs, self.hidden = self.rnn(self.packed)
        self.outputs, _ = nn.utils.rnn.pad_packed_sequence(self.outputs, batch_first=True)
        # print(outputs.shape)
        return self.outputs, self.hidden

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, 
                 embedding_size : int,
                 hidden_size : int,
                 decoder_layers : int,
                 rnn_type = "LSTM",
                 dropout = 0.2,
                 pad_idx = 0):
        
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.decoder_layers = decoder_layers
        self.rnn_type = rnn_type
        self.dropout = dropout
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(target_vocab_size, embedding_size, padding_idx=pad_idx)

        # initialize the RNN
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_size, hidden_size, decoder_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_size, hidden_size, decoder_layers, batch_first=True, dropout=dropout)
        elif self.rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_size, hidden_size, decoder_layers, batch_first=True, dropout=dropout)
        else:
            raise ValueError(f"Unsupported rnn_type: {self.rnn_type}")
        
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, target_vocab_size),
            nn.Tanh(),

            nn.Dropout(self.dropout),
            nn.Linear(target_vocab_size, target_vocab_size),
        )

        self.fc_out = nn.Linear(hidden_size, target_vocab_size)

    def forward(self, input_token, hidden):
        """
        input_token: Tensor of shape [batch_size] -> current token indices
        hidden: hidden state from previous step, shape [num_layers, batch_size, hidden_size]
        """
        input_token = input_token.unsqueeze(1)  # [batch_size, 1]
        embedded = nn.Dropout(self.dropout)(self.embedding(input_token))  # [batch_size, 1, emb_dim]

        if self.rnn_type == "LSTM":
            output, (h, c) = self.rnn(embedded, hidden) # output: [batch_size, 1, hidden_size]
            return self.fc_out(output.squeeze(1)), (h, c) # output: [batch_size, vocab_size]
        else:
            output, h = self.rnn(embedded, hidden)
            return self.fc_out(output.squeeze(1)), h

class Seq_2_Seq(nn.Module):
    def __init__(self, encoder : Encoder, 
                 decoder : Decoder, 
                 teacher_forcing_ratio : float = 1.0):
        super().__init__()

        # initializing encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn_type = self.decoder.rnn_type
    
    def forward(self, native_tensor, native_tensor_len, roman_tensor):

        # getting the hidden of the native text
        _,native_hidden = self.encoder(native_tensor, native_tensor_len)

        # outputs for storing the output from the model
        target_vocab_size = self.decoder.embedding.weight.shape[0]
        batch_size = roman_tensor.shape[0]
        target_len = roman_tensor.shape[1]
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)

        # initializing the decoder hidden state
        decoder_hidden = native_hidden

        # running the decoder
        decoder_input = roman_tensor[:, 0]
        # print(f"Decoder input shape : {decoder_input.shape}") # -> passed

        for i in range(1, target_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
            outputs[:, i] = output

            # print(f"Output shape : {output.shape}")

            # Teacher forcing with a ratio
            teacher_force = torch.rand(1) < self.teacher_forcing_ratio
            best_guess = output.argmax(1)
            decoder_input = roman_tensor[:, i] if teacher_force else best_guess

        return outputs