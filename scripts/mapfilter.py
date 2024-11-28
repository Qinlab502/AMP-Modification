#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils import *
from models import symmetrize, ContactMapRegression

def property_filter(fasta_file, device):
    lbd_dict = get_fasta_dict(fasta_file)
    seqs=[]
    for header,seq in lbd_dict.items():
        seqs.append(seq)
    predictions = get_property_predictions(seqs, device)
    index = torch.where(predictions == 0)

    return index


def binary_matrix_intersection(matrices):

    intersection_matrix = np.array(matrices[0])
    for matrix in matrices[1:]:
        intersection_matrix = np.logical_and(intersection_matrix, matrix)
    intersection_matrix = intersection_matrix.astype(int)

    return intersection_matrix


def get_intersection(fasta_file, device, checkpoint_path):
    pred_contacts = get_pred_contact(fasta_file, device, checkpoint_path)
    index = property_filter(fasta_file, device)
    filter = pred_contacts[index]
    intersection = binary_matrix_intersection(filter)
    return intersection


def site_filter(pfasta_file, nfasta_file, device, checkpoint_path):
    """
    All positive and negative contact maps are intersected to 
    filter out the sites that are not in common between the two classes.
    Refer to plan a
    """
    p_map = get_intersection(pfasta_file, device, checkpoint_path)
    n_map = get_intersection(nfasta_file, device, checkpoint_path)
    p_map_new = symmetrize(p_map)
    n_map_new = symmetrize(n_map)
    row_indices, col_indices = np.where(p_map_new != n_map_new)

    # Keep only the upper triangular indices (to avoid duplicates)
    mask = row_indices < col_indices
    filtered_row_indices = row_indices[mask]
    filtered_col_indices = col_indices[mask]
    # Filter results where the difference between row and column indices > 1
    final_mask = np.abs(filtered_row_indices - filtered_col_indices) > 1
    final_row_indices = filtered_row_indices[final_mask]
    final_col_indices = filtered_col_indices[final_mask]

    filtered_result = (final_row_indices, final_col_indices)

    return filtered_result


def regression_train(
    ContactMapRegression,
    in_features: int,
    pfasta_file,
    nfasta_file,
    device: torch.device,
    num_epochs,
    learning_rate: float = 1e-2,
    check_point_path: str = None,
    save: bool = False,
    output_model: str = None
):
    """
    Train a regression model to filter contact maps. Refer to plan b.
    """

    contactmodel = ContactMapRegression(in_features)
    contactmodel.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load inputs and labels
    p_contact = get_pred_contact(pfasta_file, device, check_point_path)
    p_index = property_filter(pfasta_file, device)
    p_contact_filer = p_contact[p_index]

    n_contact = get_pred_contact(nfasta_file, device, check_point_path)
    n_index = property_filter(nfasta_file, device)
    n_contact_filer = n_contact[n_index]

    cmin=torch.cat([p_contact_filer,n_contact_filer],dim=0)
    cmin=torch.tensor(cmin,dtype=torch.float32).to(device)
    cmin=cmin.detach()

    yp=torch.ones(len(p_contact_filer))
    yn=torch.zeros(len(n_contact_filer))
    y=torch.cat([yp,yn],dim=0)
    y=torch.tensor(y,dtype=torch.long).to(device)

    # Train model
    loss_list = []
    accurate_list = []

    for epoch in tqdm(range(num_epochs)):
        contactmodel.train()
        outputs=contactmodel(cmin)

        loss = criterion(outputs, y)
        _, prediction = outputs.max(dim=1)
        accurate = (prediction==y).sum().item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_accuracy = accurate / len(outputs)
        average_loss = loss / len(outputs)

        loss_list.append(average_loss)
        accurate_list.append(train_accuracy)
    
    if save:
        save_model(contactmodel, output_model)

    return loss_list, accurate_list

