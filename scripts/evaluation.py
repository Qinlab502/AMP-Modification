#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from typing import Optional, Dict, Union, Callable
import matplotlib as mpl

from utils import *
from contact import get_true_contacts, extract_single_letter_sequence

#Adapt from: https://github.com/facebookresearch/esm/blob/main/examples/contact_prediction.ipynb
def compute_precisions(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    src_lengths: Optional[torch.Tensor] = None,
    minsep: int = 6,
    maxsep: Optional[int] = None,
    override_length: Optional[int] = None,  # for casp
):
    if isinstance(predictions, np.ndarray):
        predictions = torch.from_numpy(predictions)
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(0)
    if targets.dim() == 2:
        targets = targets.unsqueeze(0)
    override_length = (targets[0, 0] >= 0).sum()

    # Check sizes
    if predictions.size() != targets.size():
        raise ValueError(
            f"Size mismatch. Received predictions of size {predictions.size()}, "
            f"targets of size {targets.size()}"
        )
    device = predictions.device

    batch_size, seqlen, _ = predictions.size()
    seqlen_range = torch.arange(seqlen, device=device)

    sep = seqlen_range.unsqueeze(0) - seqlen_range.unsqueeze(1)
    sep = sep.unsqueeze(0)
    valid_mask = sep >= minsep
    valid_mask = valid_mask & (targets >= 0)  # negative targets are invalid

    if maxsep is not None:
        valid_mask &= sep < maxsep

    if src_lengths is not None:
        valid = seqlen_range.unsqueeze(0) < src_lengths.unsqueeze(1)
        valid_mask &= valid.unsqueeze(1) & valid.unsqueeze(2)
    else:
        src_lengths = torch.full([batch_size], seqlen, device=device, dtype=torch.long)

    predictions = predictions.masked_fill(~valid_mask, float("-inf"))

    x_ind, y_ind = np.triu_indices(seqlen, minsep)
    predictions_upper = predictions[:, x_ind, y_ind]
    targets_upper = targets[:, x_ind, y_ind]

    topk = seqlen if override_length is None else max(seqlen, override_length)
    indices = predictions_upper.argsort(dim=-1, descending=True)[:, :topk]
    topk_targets = targets_upper[torch.arange(batch_size).unsqueeze(1), indices]
    if topk_targets.size(1) < topk:
        topk_targets = F.pad(topk_targets, [0, topk - topk_targets.size(1)])

    cumulative_dist = topk_targets.type_as(predictions).cumsum(-1)

    gather_lengths = src_lengths.unsqueeze(1)
    if override_length is not None:
        gather_lengths = override_length * torch.ones_like(
            gather_lengths, device=device
        )

    gather_indices = (
        torch.arange(0.1, 1.1, 0.1, device=device).unsqueeze(0) * gather_lengths
    ).type(torch.long) - 1

    binned_cumulative_dist = cumulative_dist.gather(1, gather_indices)
    binned_precisions = binned_cumulative_dist / (gather_indices + 1).type_as(
        binned_cumulative_dist
    )

    pl5 = binned_precisions[:, 1]
    pl2 = binned_precisions[:, 4]
    pl = binned_precisions[:, 9]
    auc = binned_precisions.mean(-1)

    return {"AUC": auc, "P@L": pl, "P@L2": pl2, "P@L5": pl5}


def evaluate_prediction(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> Dict[str, float]:
    if isinstance(targets, np.ndarray):
        targets = torch.from_numpy(targets)
    contact_ranges = [
        ("local", 3, 6),
        ("short", 6, 12),
        ("medium", 12, 24),
#         ("long", 24, None),
    ]
    metrics = {}
    targets = targets.to(predictions.device)
    for name, minsep, maxsep in contact_ranges:
        rangemetrics = compute_precisions(
            predictions,
            targets,
            minsep=minsep,
            maxsep=maxsep,
        )
        for key, val in rangemetrics.items():
            metrics[f"{name}_{key}"] = val.item()
    return metrics


# Adapted from: https://github.com/rmrao/evo/blob/main/evo/visualize.py
def plot_contacts_and_predictions(
    predictions: Union[torch.Tensor, np.ndarray],
    contacts: Union[torch.Tensor, np.ndarray],
    ax: Optional[mpl.axes.Axes] = None,
    # artists: Optional[ContactAndPredictionArtists] = None,
    cmap: str = "Blues",
    ms: float = 1,
    title: Union[bool, str, Callable[[float], str]] = True,
    animated: bool = False,
) -> None:

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(contacts, torch.Tensor):
        contacts = contacts.detach().cpu().numpy()
    if ax is None:
        ax = plt.gca()

    seqlen = contacts.shape[0]
    relative_distance = np.add.outer(-np.arange(seqlen), np.arange(seqlen))
    bottom_mask = relative_distance < 0  #形成一个三角
    masked_image = np.ma.masked_where(bottom_mask, predictions)
    invalid_mask = np.abs(np.add.outer(np.arange(seqlen), -np.arange(seqlen))) < 6
    predictions = predictions.copy()
    predictions[invalid_mask] = float("-inf")

    topl_val = np.sort(predictions.reshape(-1))[-seqlen]
    pred_contacts = predictions >= topl_val
    true_positives = contacts & pred_contacts & ~bottom_mask
    false_positives = ~contacts & pred_contacts & ~bottom_mask
    other_contacts = contacts & ~pred_contacts & ~bottom_mask

    if isinstance(title, str):
        title_text: Optional[str] = title
    elif title:
        long_range_pl = compute_precisions(predictions, contacts, minsep=24)[
            "P@L"
        ].item()
        if callable(title):
            title_text = title(long_range_pl)
        else:
            title_text = f"Long Range P@L: {100 * long_range_pl:0.1f}"
    else:
        title_text = None

    img = ax.imshow(masked_image, cmap=cmap, animated=animated)
    oc = ax.plot(*np.where(other_contacts), "o", c="grey", ms=ms)[0]
    fn = ax.plot(*np.where(false_positives), "o", c="r", ms=ms)[0]
    tp = ax.plot(*np.where(true_positives), "o", c="b", ms=ms)[0]
    ti = ax.set_title(title_text) if title_text is not None else None
    # artists = ContactAndPredictionArtists(img, oc, fn, tp, ti)

    ax.axis("square")
    ax.set_xlim([-0.5, seqlen-0.5])
    ax.set_ylim([-0.5, seqlen-0.5])
    ax.set_xticks(np.arange(1,seqlen,2))
    ax.set_yticks(np.arange(1,seqlen,2))
    plt.colorbar(img, fraction=0.045)


def evaluate_property_model(test_fasta_path, device):
    
    test_dict = get_fasta_dict(test_fasta_path)
    dataset = SeqDataset(test_dict)
    print("Dataset checking:")
    sequences = []
    labels = []
    for i in tqdm(range(len(dataset))):
        seq, label = dataset[i]
        sequences.append(seq)
        labels.append(label)
    predictions = get_property_predictions(sequences, device)
    
    # Compute confusion matrix and classification report
    confusion_mat = confusion_matrix(labels, predictions)
    report = classification_report(labels, predictions)
    print(f"Confusion Matrix:\n{confusion_mat}\n")
    print(f"Classification Report:\n{report}")

    return labels, predictions


def evaluate_contact_model(check_point_path, cif_file_path, device):
    
    # Load true contacts
    cif_file = cif_file_path
    contacts = get_true_contacts(cif_file)
    print("Loaded true contact map.")

    # Extract protein sequence
    sequence = extract_single_letter_sequence(cif_file, chain_id=None)

    # Load tokenizer and prepare inputs
    inputs = input_token(sequence, max_length, tokenizer, device)
    lora_model=load_lora_model(model_path, lora_path, device=device)
    with torch.no_grad():
        outputs = lora_model.esm(**inputs, output_attentions=True, output_hidden_states=True)
        lbd_lora_attention = torch.stack(outputs.attentions, 1)
    mymodel = load_model_checkpoint(check_point_path, device)
    
    #contact predicts from lora_esm2
    lbd_lora_out=lora_model.esm.contact_head(inputs['input_ids'],lbd_lora_attention)
    fig, axes = plt.subplots(figsize=(6, 6))
    plot_contacts_and_predictions(
            lbd_lora_out[0], contacts, ax=axes, title = 'LBDB_prediction'
        )
    plt.show()

    metrics = {"id": 'lbdb', "model": "ESM-2 (Unsupervised)"}
    metrics.update(evaluate_prediction(lbd_lora_out, contacts))
    print(metrics)

    #contact predicts from contact-based model
    new_predictions=mymodel(inputs['input_ids'], lbd_lora_attention)[0].cpu()
    fig, axes = plt.subplots(figsize=(6, 6))
    plot_contacts_and_predictions(
            new_predictions, contacts, ax=axes, title = 'LBDB_prediction'
        )
    plt.show()

    metrics = {"id": 'lbdb', "model": "ESM-2 (Unsupervised)"}
    metrics.update(evaluate_prediction(new_predictions, contacts))
    print(metrics)


def evaluate_contactmapregression_model(model_path, check_point_path, fasta_file, labels, device):
    
    model=load_model_checkpoint(model_path, device)
    
    #get contact maps
    pred_contact = get_pred_contact(fasta_file, device, check_point_path)
    pred_contact=torch.tensor(pred_contact,dtype=torch.float32).to(device)
    #get predictions
    predictions=model(pred_contact)
    _,prediction=predictions.max(dim=1)
    prediction=prediction.cpu()

    labels=torch.tensor(labels,dtype=torch.long).numpy()
    # Compute confusion matrix and classification report
    confusion_mat = confusion_matrix(labels, prediction.cpu())
    report = classification_report(prediction.cpu(), labels)
    print(f"Confusion matrix: \n{confusion_mat}")
    print(report)
