import torch
from torch.utils.data import Dataset, DataLoader

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

def word_to_vector(word, char2idx, max_len = None, sos=False, eos=False):

    tokens = []
    if sos: tokens.append('<sos>')
    tokens.extend(list(word))
    if eos: tokens.append('<eos>')

    if max_len != None and len(tokens) > max_len:
        tokens = tokens[:max_len]
    elif max_len != None and len(tokens) < max_len:
        tokens = tokens + ['<pad>'] * (max_len - len(tokens))

    tensor = torch.zeros(len(tokens))
    for i, char in enumerate(tokens):
        idx = char2idx.get(char, char2idx['<unk>'])
        tensor[i] = idx
    return tensor

class Dakshina_Dataset(Dataset):
    def __init__(self, data, 
                 native_char_to_ind, roman_char_to_ind, 
                 max_len_native = None, max_len_roman = None):

        self.data = data
        self.native_char_to_ind = native_char_to_ind
        self.roman_char_to_ind = roman_char_to_ind
        self.max_len_native = max_len_native
        self.max_len_roman = max_len_roman
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        native, roman = self.data[index]

        # one-hot encoding of the chars in native
        native_tensor = word_to_vector(native, self.native_char_to_ind,
                                              max_len = self.max_len_native if self.max_len_native != None else len(native),
                                              sos = False, eos = False).to(torch.long)
        native_input_length = len(native)

        # one-hot encoding of the chars in roman (it need sos and eos tokens for start and end of word indication)
        roman_tensor = word_to_vector(roman, self.roman_char_to_ind,
                                             max_len = self.max_len_roman if self.max_len_roman != None else len(roman),
                                             sos = True, eos = True).to(torch.long)
        roman_input_length = len(roman) + 2

        return (native_tensor, native_input_length, roman_tensor, roman_input_length)

def get_loaders(train_df, val_df, test_df, native_vocab, roman_vocab, batch_size=512):
    train_dataset = Dakshina_Dataset(train_df.values, native_vocab, roman_vocab)
    val_dataset = Dakshina_Dataset(val_df.values, native_vocab, roman_vocab)
    test_dataset = Dakshina_Dataset(test_df.values, native_vocab, roman_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
