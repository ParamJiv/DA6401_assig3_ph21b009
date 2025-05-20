import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" Attention Network """

class Attention_module(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super(Attention_module, self).__init__()
        self.attn = nn.Linear(encoder_hidden_dim + decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Parameter(torch.rand(decoder_hidden_dim))

    def forward(self, hidden, encoder_outputs, src_lengths):
        # hidden: [batch_size, hidden_dim]
        # encoder_outputs: [batch_size, src_len, hidden_dim]
        
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden[-1]

        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)  # [batch_size, src_len, hidden_dim]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [batch_size, src_len, hidden_dim]
        energy = torch.einsum("bsh,h->bs", energy, self.v)  # [batch_size, src_len]

        # Mask padding
        mask = torch.arange(src_len).expand(len(src_lengths), src_len).to(src_lengths.device) >= src_lengths.unsqueeze(1)
        mask = mask.to(device)
        energy = energy.masked_fill(mask, -1e10)

        attention_weights = F.softmax(energy, dim=1)  # [batch_size, src_len]

        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch_size, 1, hidden_dim]
        del mask, energy
        return context, attention_weights

class Encoder_for_attention(nn.Module):
    def __init__(self, input_vocab_size, 
                 embedding_size : int ,
                 hidden_size : int, 
                 encoder_layers : int = 1,
                 rnn_type = "LSTM",
                 dropout = 0.2,
                 pad_idx = 0 ):
        
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
        embedded = nn.Dropout(self.dropout)(self.embedding(input_idxs))  # shape: [batch, seq_len, emb_dim]
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False) # for avoiding paddings
        outputs, hidden = self.rnn(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        # print(output.shape)
        return hidden, outputs

class Decoder_with_attention(nn.Module):
    def __init__(self, output_dim, embedding_size, hidden_size, num_layers=1,
                 dropout=0.1, rnn_type='LSTM', pad_idx=None, attention=None):
        super(Decoder_with_attention, self).__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type.upper()
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, embedding_size, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        rnn_cls = {
            'RNN': nn.RNN,
            'GRU': nn.GRU,
            'LSTM': nn.LSTM
        }[self.rnn_type]

        # RNN takes attention context + embedding as input
        self.rnn = rnn_cls(embedding_size + hidden_size, hidden_size, num_layers, batch_first=True, dropout = dropout)

        # Final output layer to map hidden state to vocab
        self.fc_out = nn.Linear(hidden_size * 2, output_dim)

    def forward(self, input_token, hidden, encoder_outputs, src_lengths):
        """
        input_token: (batch_size)
        hidden: (num_layers, batch_size, hidden_size) for RNN/GRU or tuple for LSTM
        encoder_outputs: (batch_size, src_len, hidden_size)
        src_lengths: (batch_size,) actual lengths before padding (needed for attention)
        """
        # Embed input token
        embedded = self.dropout(self.embedding(input_token))  # (batch_size, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch_size, 1, embed_dim)

        # Attention context vector: (batch_size, 1, hidden_size)
        if self.attention is not None:
            if self.rnn_type == "LSTM": context, _ = self.attention(hidden[0], encoder_outputs, src_lengths)  # (batch_size, 1, hidden_dim)
            else: context, _ = self.attention(hidden, encoder_outputs, src_lengths)
        else:
            context = torch.zeros(embedded.size(0), 1, self.hidden_dim, device=embedded.device)

        # Combine context and embedded input
        # print(context.shape, embedded.shape)
        rnn_input = torch.cat((embedded, context), dim=2)  # (batch_size, 1, embed + hidden)

        # Handle hidden state format
        if self.rnn_type == "LSTM":
            # print(rnn_input.shape, context.shape)
            output, (h, c) = self.rnn(rnn_input, hidden)  # output: (batch, 1, hidden)
            new_hidden = (h, c)
        else:
            output, h = self.rnn(rnn_input, hidden)
            new_hidden = h

        # Output is (batch, 1, hidden), context is (batch, 1, hidden)
        combined = torch.cat((output, context), dim=2)  # (batch, 1, 2*hidden)
        prediction = self.fc_out(combined.squeeze(1))   # (batch, output_dim)

        return prediction, new_hidden
class Seq_2_Seq_with_attention(nn.Module):
    def __init__(self, encoder , 
                 decoder , 
                 teacher_forcing_ratio : float = 1.0):
        super().__init__()

        # initializing encoder and decoder
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.rnn_type = self.decoder.rnn_type
    
    def forward(self, native_tensor, native_tensor_len, roman_tensor):

        # getting the hidden of the native text
        native_hidden, native_outputs = self.encoder(native_tensor, native_tensor_len)

        # outputs for storing the output from the model
        target_vocab_size = self.decoder.embedding.weight.shape[0]
        batch_size = roman_tensor.shape[0]
        target_len = roman_tensor.shape[1]
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(device)

        # initializing the decoder hidden state
        decoder_hidden = native_hidden
        # print(native_outputs.data.shape)

        # running the decoder
        decoder_input = roman_tensor[:, 0]
        # print(f"Decoder input shape : {decoder_input.shape}") # -> passed

        for i in range(1, target_len):
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, 
                                                encoder_outputs = native_outputs, 
                                                src_lengths = native_tensor_len)
            outputs[:, i] = output

            # Teacher forcing with a ratio
            teacher_force = torch.rand(1) < self.teacher_forcing_ratio
            best_guess = output.argmax(1)
            decoder_input = roman_tensor[:, i] if teacher_force else best_guess

        return outputs