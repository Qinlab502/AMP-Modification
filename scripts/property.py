#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
property.py

This script can help you train a property prediction model.

Example:
    python property.py --model_path ../models/esm2_650M -i ../database/LBD_135.fasta
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.optim import AdamW
from transformers import get_scheduler
from peft import get_peft_model, LoraConfig, TaskType
from focal_loss.focal_loss import FocalLoss

import argparse
from collections import Counter
from utils import *

def prepare_dataloaders(fasta_path, tokenizer, max_length, batch_size, random_seed=42, train_split=0.8):
    fasta_dict = get_fasta_dict(fasta_path)
    dataset = SeqDataset(fasta_dict)
    print("Dataset checking:")
    for i in tqdm(range(len(dataset))):
        seqs, label = dataset[i]

    # Count label distribution
    labels = get_label(fasta_dict)
    label_counts = Counter(labels)
    print(f"Label distribution: {label_counts}")

    # Set random seed and shuffle data
    np.random.seed(random_seed)
    indices = np.random.permutation(len(dataset))
    train_size = int(len(dataset) * train_split)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # Create training and testing datasets
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    return train_loader, test_loader, label_counts

def initialize_lora(model, task_type, r, alpha, dropout, target_modules, modules_to_save):
    """
    Initialize LoRA configuration and return the modified model.

    Args:
        model: Original model.
        task_type (TaskType): Task type.
        r (int): LoRA rank.
        alpha (int): LoRA alpha parameter.
        dropout (float): LoRA dropout probability.
        target_modules (list): List of module names to apply LoRA.
        modules_to_save (list): List of submodule names to save.

    Returns:
        lora_model
    """
    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    lora_model = get_peft_model(model, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model

def train_model(
    model,
    train_loader,
    test_loader,
    tokenizer,
    device,
    max_length,
    num_epochs,
    lr,
    weight_decay,
    progress_bar_description="Training"
):

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    focal_loss = FocalLoss(gamma=1)

    train_acc_list = []
    test_acc_list = []
    train_loss_list = []
    test_loss_list = []

    progress_bar = tqdm(range(num_training_steps), desc=progress_bar_description)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_correct = 0
        train_total = 0
        epoch_train_loss = 0.0

        for batch in train_loader:
            sequences, labels = batch
            optimizer.zero_grad()

            inputs = tokenizer(
                sequences,
                return_tensors='pt',
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)
            labels = labels.to(device)

            outputs = model(**inputs, labels=labels)
            logits = outputs.logits
            loss = focal_loss(torch.softmax(logits, dim=1), labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # Calculate training accuracy
            predictions = torch.argmax(logits, dim=1)
            train_total += labels.size(0)
            train_correct += (predictions == labels).sum().item()
            epoch_train_loss += loss.item()

        train_acc = train_correct / train_total
        epoch_train_loss /= len(train_loader)
        train_acc_list.append(train_acc)
        train_loss_list.append(epoch_train_loss)

        # Validation phase
        model.eval()
        test_correct = 0
        test_total = 0
        epoch_test_loss = 0.0

        with torch.no_grad():
            for batch in test_loader:
                sequences, labels = batch
                inputs = tokenizer(
                    sequences,
                    return_tensors='pt',
                    padding="max_length",
                    truncation=True,
                    max_length=max_length
                ).to(device)
                labels = labels.to(device)

                outputs = model(**inputs, labels=labels)
                logits = outputs.logits
                loss = focal_loss(torch.softmax(logits, dim=1), labels)
                predictions = torch.argmax(logits, dim=1)
                test_total += labels.size(0)
                test_correct += (predictions == labels).sum().item()
                epoch_test_loss += loss.item()

        test_acc = test_correct / test_total
        epoch_test_loss /= len(test_loader)
        test_acc_list.append(test_acc)
        test_loss_list.append(epoch_test_loss)

        print(f"Epoch: {epoch}, "
              f"Training Loss: {epoch_train_loss:.2f}, "
              f"Test Loss: {epoch_test_loss:.2f}, "
              f"Training Accuracy: {train_acc:.4f}, "
              f"Test Accuracy: {test_acc:.4f}")

    return train_acc_list, test_acc_list, train_loss_list, test_loss_list

def save_model(model, model_name, peft_type, task_type, test_acc):
    peft_model_id = f"{model_name}_{peft_type}_{task_type}_{test_acc:.2f}"
    model.save_pretrained(peft_model_id)
    print(f"Model saved to {peft_model_id}")

def parse_arguments():

    parser = argparse.ArgumentParser(description="Traning process for property prediction.")

    # Model and Data Paths
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the pre-trained ESM2 model.')
    parser.add_argument('-i', '--train_fasta_path', type=str, required=True,
                        help='Path to the training FASTA file.')
    # Hyperparameters
    parser.add_argument('--max_length', type=int, default=24,
                        help='Maximum sequence length.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate.')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of classification classes.')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--weight_decay', type=float, default=5e-3,
                        help='Weight decay for optimizer.')
    # LoRA Parameters
    parser.add_argument('-r', '--rank', type=int, default=48,
                        help='LoRA rank.')
    parser.add_argument('-a', '--alpha', type=int, default=24,
                        help='LoRA alpha parameter.')
    parser.add_argument('-d', '--dropout', type=float, default=0.6,
                        help='LoRA dropout probability.')
    parser.add_argument('--target_modules', nargs='+', default=["query", "value", "key"],
                        help='List of target modules to apply LoRA.')
    parser.add_argument('--modules_to_save', nargs='+', default=["decode_head"],
                        help='List of modules to save.')
    # Output Paths
    parser.add_argument('--save_model', action='store_true',
                        help='Flag to save the trained model.')
    parser.add_argument('--model_save_path', type=str, default=None,
                        help='Path to save the trained model.')

    args = parser.parse_args()
    return args

def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Set parameters from arguments
    model_path = args.model_path
    train_fasta_path = args.train_fasta_path
    max_length = args.max_length
    lr = args.lr
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    num_classes = args.num_classes
    random_seed = args.random_seed
    weight_decay = args.weight_decay

    # Set device
    device = set_device()

    # Load model and tokenizer
    tokenizer, model = load_model(model_path, num_classes, device)

    # Prepare datasets and dataloaders
    train_loader, test_loader, label_counts = prepare_dataloaders(
        fasta_path=train_fasta_path,
        tokenizer=tokenizer,
        max_length=max_length,
        batch_size=batch_size,
        random_seed=random_seed,
        train_split=0.8
    )

    # Initialize LoRA model
    lora_model = initialize_lora(
        model=model,
        task_type=TaskType.SEQ_CLS,
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=args.target_modules,
        modules_to_save=args.modules_to_save
    )

    # Train the model
    train_acc_list, test_acc_list, train_loss_list, test_loss_list = train_model(
        model=lora_model,
        train_loader=train_loader,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        max_length=max_length,
        num_epochs=num_epochs,
        lr=lr,
        weight_decay=weight_decay,
        progress_bar_description="Training"
    )

    #Save the model
    if args.save_model:
        if args.model_save_path:
            model_path = args.model_save_path
            save_model(
                model=lora_model,
                model_name=model_path,
                peft_type="LORA",
                task_type="SEQ_CLS",
                test_acc=test_acc_list[-1]
            )

if __name__ == "__main__":
    main()