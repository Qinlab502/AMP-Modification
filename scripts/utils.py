#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import EsmTokenizer, EsmForSequenceClassification
from peft import PeftModel, PeftConfig
import numpy as np

model_path = '../models/esm2_650M'
lora_path = '../models/esm2_650M_LORA_SEQ_CLS_0.99'
tokenizer = EsmTokenizer.from_pretrained(model_path)
max_length = 24

# Define amino acid mapping and hydrophobicity values
amino_acid = {
    4: 'L', 
    5: 'A', 
    6: 'G', 
    7: 'V', 
    8: 'S', 
    9: 'E', 
    10: 'R', 
    11: 'T', 
    12: 'I', 
    13: 'D',
    14: 'P', 
    15: 'K', 
    16: 'Q', 
    17: 'N', 
    18: 'F', 
    19: 'Y', 
    20: 'M', 
    21: 'H', 
    22: 'W', 
    23: 'C'
}

hydrophobicity = {
    4: 1.700,
    5: 0.310,
    6: 0.0,
    7: 1.220,
    8: -0.040,
    9: -0.640,
    10: -1.010,
    11: 0.260,
    12: 1.800,
    13: -0.770,
    14: 0.720,
    15: -0.990,
    16: -0.220,
    17: -0.600,
    18: 1.790,
    19: 0.960,
    20: 1.230,
    21: 0.130,
    22: 2.250,
    23: 1.540
}

three_to_one = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
    'CYX': 'C', 'ASH': 'D', 'GLH': 'E', 'HID': 'H',
    # Add any non-standard residues if necessary
}

# Create a dictionary mapping amino acids to their corresponding keys
dic_new = {aa: key for key, aa in amino_acid.items()}

def parse_fasta(file_path):
    """
    Parses a FASTA file and returns a dictionary with sequence names as keys and sequences as values.
    """
    sequences = {}
    with open(file_path, 'r') as f:
        name = ""
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                name = line.split()[0][1:]
                sequences[name] = ''
            else:
                sequences[name] += line.replace('\n', '')
    return sequences

def get_net_charge(sequence):

    charge = 0.0
    positive_residues = ['K', 'R', 'H']
    negative_residues = ['D', 'E']
    for res in sequence:
        if res in positive_residues:
            charge += 1.0
        elif res in negative_residues:
            charge -= 1.0
    return charge

def get_mean_hydrophobicity(sequence):

    total_hydro = 0.0
    count = 0
    for res in sequence:
        if res in dic_new:
            key = dic_new[res]
            hydro = hydrophobicity.get(key, 0.0)
            total_hydro += hydro
            count += 1
        else:
            # Amino acids not in the hydrophobicity dictionary are skipped
            continue
    if count == 0:
        return 0.0
    return total_hydro / count

def get_label(fasta_dict):
    """
    Generates binary labels for each sequence based on net charge and mean hydrophobicity.

    Labeling Criteria:
        - If 2.0 <= net charge <= 7.0 and 0.5 <= mean hydrophobicity <= 0.7, label = 0
        - Else, label = 1
    """
    label = []
    for header, sequence in fasta_dict.items():
        charge = get_net_charge(sequence)
        hy = get_mean_hydrophobicity(sequence)
        if 2.0 <= charge <= 7.0 and 0.5 <= hy <= 0.7:
            label.append(0)
        else:
            label.append(1)
    return label

class SeqDataset(Dataset):
    def __init__(self, fasta_dict):
        super().__init__()
        self.fasta_dict = fasta_dict
        self.names = list(fasta_dict.keys())
        self.labels = get_label(fasta_dict)
        
    def __getitem__(self, idx):
        seq_name = self.names[idx]
        selected_seq = self.fasta_dict[seq_name]
        label = self.labels[idx]

        return selected_seq, label
        
    def __len__(self):
        return len(self.names)

def get_fasta_dict(fasta_file):
    return parse_fasta(fasta_file)

def get_sequence(fasta_file):
    fasta_dict = get_fasta_dict(fasta_file)
    sequences = []
    for header, sequence in fasta_dict.items():
        sequences.append(sequence)
    return sequences

def set_device() -> torch.device:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device

def load_model(model_path, num_classes, device):
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=num_classes).to(device)
    return tokenizer, model

def load_model_checkpoint(checkpoint_path: str, device: torch.device = torch.device('cpu')) -> nn.Module:
    model = torch.load(checkpoint_path)
    model.eval()
    model.to(device)
    return model

def load_lora_model(model_path, lora_path, device):
    config = PeftConfig.from_pretrained(lora_path)
    base_model = EsmForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    model.to(device)
    return model

def save_model(model: nn.Module, save_path: str) -> None:
    torch.save(model, save_path)

def input_token(sequence, max_length, tokenizer, device):
    inputs = tokenizer(
        sequence,
        return_tensors='pt',
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)
    return inputs

def get_property_predictions(sequences, device):

    tokens = input_token(sequences, max_length, tokenizer, device)
    model=load_lora_model(model_path, lora_path, device)
    with torch.no_grad():
        outputs = model(**tokens)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1).cpu()

    return predictions

def get_contacts_predictions(check_point_path, sequence, device):
    
    # Load lora_model and prepare inputs
    inputs = input_token(sequence, max_length, tokenizer, device)
    lora_model=load_lora_model(model_path, lora_path, device=device)

    # Get attentions from lora model
    with torch.no_grad():
        outputs = lora_model.esm(**inputs, output_attentions=True, output_hidden_states=True)
        lbd_lora_attention = torch.stack(outputs.attentions, 1)
    
    # load contact-based-model checkpoint
    mymodel = load_model_checkpoint(check_point_path, device)
    predictions = mymodel(inputs['input_ids'], lbd_lora_attention)[0].cpu()

    return predictions

def get_pred_contact(fasta_file, device, check_point_path):
    sequence = get_sequence(fasta_file)
    attention = get_contacts_predictions(check_point_path, sequence, device)
    contact=torch.where(attention < 0.9, torch.tensor(0), torch.tensor(1))
    return contact