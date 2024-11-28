#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
contact.py

This script can help you train a contact prediction model.

Example:
    python contact.py -i ../database/lbdb.cif
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.distance import squareform, pdist

import biotite.structure as bs
from biotite.structure.io.pdbx import CIFFile, get_structure
from tqdm import tqdm
import pandas as pd
import torch.nn.functional as F

from transformers import EsmTokenizer, get_scheduler
from typing import List, Tuple, Optional, Dict
from models import *
from utils import *


#Adapted from: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
def extend(a: np.ndarray, b: np.ndarray, c: np.ndarray, L: float, A: float, D: float) -> np.ndarray:
    """
    input:  3 coords (a,b,c), (L)ength, (A)ngle, and (D)ihedral
    output: 4th coord
    """
    def normalize(x):
        return x / np.linalg.norm(x, ord=2, axis=-1, keepdims=True)

    bc = normalize(b - c)
    n = normalize(np.cross(b - a, bc))
    m = [bc, np.cross(n, bc), n]
    d = [L * np.cos(A), L * np.sin(A) * np.cos(D), -L * np.sin(A) * np.sin(D)]
    return c + sum([m_i * d_i for m_i, d_i in zip(m, d)])


def contacts_from_pdb(
    structure: bs.AtomArray,
    distance_threshold: float = 8.0,
    chain: Optional[str] = None,
) -> np.ndarray:
    mask = ~structure.hetero
    if chain is not None:
        mask &= structure.chain_id == chain

    N = structure.coord[mask & (structure.atom_name == "N")]
    CA = structure.coord[mask & (structure.atom_name == "CA")]
    C = structure.coord[mask & (structure.atom_name == "C")]

    Cbeta = extend(C, N, CA, 1.522, 1.927, -2.143)
    dist = squareform(pdist(Cbeta))

    contacts = dist < distance_threshold
    contacts = contacts.astype(np.int64)
    contacts[np.isnan(dist)] = -1
    return contacts

def extract_single_letter_sequence(cif_file_path: str, chain_id: Optional[str] = None) -> str:
    """
    Extracts the protein sequence from an input CIF file and converts it to single-letter format.
    """
    try:
        structure = get_structure(CIFFile.read(cif_file_path))[0]
    except Exception as e:
        print(f"Failed to read CIF file: {e}")
        return ""

    # If a specific chain ID is provided, extract only that chain
    if chain_id:
        mask = (structure.chain_id == chain_id) & (~structure.hetero)
    else:
        mask = ~structure.hetero  # Extract all standard amino acid residues
    
    res_ids = structure.res_id[mask]
    residue_names = structure.res_name[mask]
    if len(residue_names) == 0:
        print("No valid amino acid residues found.")
        return ""

    # Identify unique residue IDs and corresponding residue names
    unique_res_ids, unique_indices = np.unique(res_ids, return_index=True)
    unique_residue_names = residue_names[unique_indices]
    
    # Convert three-letter codes to single-letter codes
    single_letter_seq = ''.join([three_to_one.get(res, 'X') for res in unique_residue_names])

    return single_letter_seq

def get_true_contacts(cif_file_path: str) -> np.ndarray:
    """
    Load the cif file and compute the true contact map.
    """
    structure = get_structure(CIFFile.read(cif_file_path))[0]
    contacts = contacts_from_pdb(structure)
    return contacts


def initialize_model(
    num_layers: int = 33,
    attention_heads: int = 20,
) -> nn.Module:
    mymodel = AttentionLogisticRegression(
        in_features=num_layers * attention_heads,
        prepend_bos=True,
        append_eos=True,
        eos_idx=2
    )

    return mymodel


def train_model(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    contacts: torch.Tensor,
    attention: torch.Tensor,
    device: torch.device,
    num_epochs,
    learning_rate: float = 5e-3
) -> Tuple[List[float], List[float]]:

    model.to(device)

    lbd_lora_attention = attention.detach()
    contact_y = contacts.float().to(device)

    # Define loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs
    )

    loss_list = []
    accurate_list = []

    for epoch in tqdm(range(num_epochs)):
        model.train()
        outputs = model(inputs['input_ids'], lbd_lora_attention)

        total_loss = 0.0
        accurate = 0

        for index, one_contact_head in enumerate(outputs):
            loss = criterion(one_contact_head, contact_y)
            total_loss += loss.item()
            predictions = (one_contact_head >= 0.9).float()
            accurate += (predictions == contact_y).sum().item() / (one_contact_head.numel())

            loss.backward()  # Calculate loss
            optimizer.step()  # Update gradient
            lr_scheduler.step()  # Update learning rate
            optimizer.zero_grad()  # Reset gradient

        train_accuracy = accurate / len(outputs)
        average_loss = total_loss / len(outputs)

        loss_list.append(average_loss)
        accurate_list.append(train_accuracy)

    return loss_list, accurate_list

def parse_arguments():
    parser = argparse.ArgumentParser(description="Training process for contact prediction.")

    parser.add_argument('-i', '--input_pdb', type=str, required=True,
                        help="Path to the input pdb file.")
    parser.add_argument('-o', '--output_model', type=str,
                        help='Path to save the trained model. Required if --save_model is set.')
    parser.add_argument('--max_length', type=int, default=24,
                        help='Maximum sequence length.')
    parser.add_argument('--num_epoch', type=int, default=6000,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--save_model', action='store_true',
                        help='Flag to save the trained model.')
    
    # Parse initial arguments
    args = parser.parse_args()

    # Conditional check: If save_model is True, output_model must be provided
    if args.save_model and not args.output_model:
        parser.error("--output_model is required when --save_model is set.")

    return args

def main():

    args = parse_arguments()

    # Set device
    device = set_device()
    print(f"Using device: {device}")

    # Load true contacts
    cif_file = args.input_pdb
    contacts = get_true_contacts(cif_file)
    print("Loaded true contact map.")

    # Extract protein sequence
    sequence = extract_single_letter_sequence(cif_file, chain_id=None)

    # Initialize model
    model = initialize_model()
    print("Initialized model.")

    # Prepare inputs
    inputs = input_token(sequence, args.max_length, tokenizer, device)

    # Obtain attention from the base model
    lora_model=load_lora_model(model_path, lora_path=lora_path, device=device)
    with torch.no_grad():
        outputs = lora_model.esm(**inputs, output_attentions=True, output_hidden_states=True)
        lbd_lora_attention = torch.stack(outputs.attentions, 1)

    # Finetune logits
    loss_list, accurate_list = train_model(
        model=model,
        inputs=inputs,
        contacts=torch.tensor(contacts, dtype=torch.float32),
        attention=lbd_lora_attention,
        device=device,
        num_epochs=args.num_epoch,
        learning_rate=args.lr
    )
    print(loss_list[-1], accurate_list[-1])
    
    # Save the trained model
    if args.save_model:
        save_model(model, args.output_model)


if __name__ == "__main__":
    main()