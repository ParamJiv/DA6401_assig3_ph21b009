# DA6401_assig3_ph21b009
# Name : ShivaSurya | Roll no : PH21B009
# Wandb report link : [Link](https://wandb.ai/shivasurya-iit-madras/DA6401-assig-3/reports/PH21B009-ShivaSurya-Assignment-3--VmlldzoxMjg0MTUzMQ?accessToken=15dpli97i1ozzt0nmb9gg41yp1siji9vxmut5lujqskvnu5pkhbjup44koisg82h)

## Kaggle notebook link : [Link](https://www.kaggle.com/code/xfactorb/da6401-assig3-nb) (Used for training)

# DA6401 Assignment 3 - Neural Machine Translation (NMT)

This repository contains the implementation of a Neural Machine Translation (NMT) model for transliteration of Tamil text using various architectures like LSTM, GRU and attention mechanisms. The project is structured to facilitate training, evaluation, and inference tasks.

## Project Structure

```plaintext
DA6401_assig3_ph21b009/
├── connectivity.py           # Visualizes attention and connectivity graphs
├── data_loader.py           # Handles dataset loading and preprocessing
├── da6401-assig3-nb.ipynb   # Jupyter Notebook for experimentation
├── hyper-parameters.yaml    # Hyperparameter grid for W&B sweeps
├── models/                  # Model architectures
│   ├── encoder.py           # Encoder model definition
│   └── decoder.py           # Decoder model definition
├── params.yaml              # Configuration parameters
├── predict.py               # Inference script
├── train.py                 # Training script
├── trainer.py               # Trainer class for model training
├── vocab/                   # Vocabulary files
├── wandb/                   # W&B logging configurations
├── weights/                 # Pre-trained model weights
└── dataset/                 # Dataset directory
    └── dakshina_dataset_v1.0/
        └── ta/
            └── lexicons/    # Lexicon files for Tamil language
```

---
| File/Folder                | Description  
|----------------------------|---------------------------------------------------------------------------------------------|
| `data_loader.py`           | Contains the `Dakshina_Dataset` class and functions for creating PyTorch DataLoaders.       |
| `load_data.py`             | Loads raw CSV data, builds vocabularies, and handles preprocessing for model input.         |
| `models/vanilla.py`        | Implements the vanilla seq2seq model: Encoder, Decoder, and Seq2Seq wrapper (no attention). |
| `models/attention.py`      | Implements the attention-based seq2seq model: Attention module, Encoder, Decoder, wrapper.  |
| `trainer.py`               | Contains the training and validation logic, including Weights & Biases (wandb) logging.     |
| `train.py`                 | Main script to launch training. Uses argparse for config, reads params from `params.yaml`.  |
| `predict.py`               | Script for inference: loads model weights, runs predictions, saves output to CSV.           |
| `params.yaml`              | YAML file specifying all hyperparameters and model configuration.                           |
| `data/`                    | Folder for train, validation, and test CSV files.                                           |
| `wandb/`                   | (Auto-generated) Directory where wandb logs and run outputs are stored.                     |

---

## How to Run the Scripts

### Training

To train a model, use the following command from your terminal:


**Command-line arguments for `train.py`:**
- `--train_path` : Path to training lexicon file

- `--val_path` : Path to validation lexicon file

- `--test_path` : Path to test file

- `--params_path` : Path to YAML file with hyperparameters (default: `params.yaml`)

- `--wandb_api_key` : Your Weights & Biases API key (**required**)

- `--wandb_project` : Name of the wandb project (default: `transliteration`)

- `--use_attention` : Add this flag to use the attention-based model

### Prediction

After training, generate predictions on the test set using:

**Command-line arguments for `predict.py`:**
- `--test_path` : Path to test file

- `--s2s_weights` : Path to the trained s2s model weights (.pth)

- `--params_path` : Path to YAML file with hyperparameters

- `--native_vocab_path` : Path to pickled native vocabulary

- `--roman_vocab_path` : Path to pickled roman vocabulary

- `--roman_idx2char_path` : Path to pickled roman idx2char 
mapping

- `--output_path` : Path to save predictions CSV (default: `predictions.csv`)

- `--use_attention` : Add this flag if using the attention-based model

---

## Model Connectivity

The project uses an encoder-decoder architecture for sequence-to-sequence learning. The encoder processes the input Tamil sequence and compresses it into a context vector, which the decoder uses to generate the Romanized sequence.

Here is an image of characters focused by the decoder at each output step (each output step - y axis - denoted by its corresponding output | x axis - Tamil Input)

<p align="center">
  <img src="connectivity images\lstm_connectivity_heatmap_அகத்தீஸ்வரர்.png" width="500"/>
</p>

Tamil word : ```அகத்தீஸ்வரர்```

---

## Attention Heatmap

In the attention-based model, the decoder at each step computes a weighted sum over encoder outputs, focusing on relevant input tokens. The attention weights can be visualized as a heatmap, where each row is a decoder output step and each column is an encoder input token. Brighter colors indicate higher attention.

<p align="center">
  <img src="attention heatmaps images\attention_heatmap_ஃபோர்ஸ்.png" width="500"/>
</p>

Tamil word : `ஃபோர்ஸ்`

---