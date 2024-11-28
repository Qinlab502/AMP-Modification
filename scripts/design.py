#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
design.py

This script is used to design LBD sequences.

Example:
    python design.py -i ../database/LBD.fasta -o ../outputs/LBD_mutation.fasta
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Tuple
import time

from utils import *
from contact import extract_single_letter_sequence


class GeneticAlgorithm():
    def __init__(
        self,
        model,
        tokenizer,
        device: str,
        seq_ids: List[int],
        pop_size: Tuple[int, int] = (None, None),
        num_parents_mating: int = 8,
        num_generations: int = 10000,
        num_mutations: int = 3
    ):
        """Initial genetic algorithm."""
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seq_ids = seq_ids
        self.pop_size = pop_size
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.num_mutations = num_mutations
        self.pp_in = np.zeros((1, len(seq_ids)))
        self.amino_acid = amino_acid

    def add(self, pop: np.ndarray) -> np.ndarray:
        """Add special tokens and Cys token to the population."""
        new_population = []
        for i in pop:
            temp = np.insert(i, 0, [0, 23])
            temp = np.insert(temp, len(temp), [23, 2])
            new_population.append(temp)
        return np.array(new_population)

    def crossover(self, parents: np.ndarray, offspring_size: Tuple[int, int]) -> np.ndarray:
        """Perform crossover between parents to generate offspring."""
        offspring = np.empty(offspring_size)
        # The point at which crossover takes place between two parents. Usually, it is at the center.
        crossover_point = np.uint8(offspring_size[1]/2)

        for k in range(offspring_size[0]):
            # Index of the first parent to mate.
            parent1_idx = k%parents.shape[0]
            # Index of the second parent to mate.
            parent2_idx = (k+1)%parents.shape[0]
            # The new offspring will have its first half of its genes taken from the first parent.
            offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
            # The new offspring will have its second half of its genes taken from the second parent.
            offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
        return offspring

    def mutation(self, offspring_crossover: np.ndarray) -> np.ndarray:
        """Introduce mutations into the offspring population."""
        if self.seq_ids is None:
            raise ValueError("seq_ids must be provided for mutation.")

        mutations_counter = np.uint8(offspring_crossover.shape[1] / self.num_mutations)
        # Mutation changes a number of genes as defined by the num_mutations argument. The changes are random.
        for idx in range(offspring_crossover.shape[0]):
            gene_idx = mutations_counter - 1
            for mutation_num in range(self.num_mutations):
                # The random value to be added to the gene.
                random_value = np.random.randint(min(self.seq_ids), max(self.seq_ids), 1)
                offspring_crossover[idx, gene_idx] = random_value
                gene_idx += mutations_counter
        return offspring_crossover

    def filter_func(self, x: np.ndarray) -> np.ndarray:
        """
        Use the fine-tuned model to get the index of the physicochemistry properties in changes
        First, wrap the input tokens
        Then, use the fine-tuned model to predict the physicochemistry properties
        Finally, filter the in distribution embeddings and return the index
        """
        
        new_population = self.add(x)
        #Wrap the input tokens
        attention_mask = np.ones((new_population.shape[0], new_population.shape[1]))
        attention_mask = attention_mask.astype(int)
        new_population_tensor = torch.tensor(new_population).to(self.device)
        attention_mask_tensor = torch.tensor(attention_mask).to(self.device)
        inputs = {'input_ids': new_population_tensor, 'attention_mask': attention_mask_tensor}

        #Extract physicochemical properties in embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1).cpu().numpy()

        index = np.where(predictions == 0)[0]
        return index

    def run(self, fasta_file) -> np.ndarray:
        """Run the genetic algorithm."""
        initial_seqs = get_sequence(fasta_file)
        initial_t = self.tokenizer(initial_seqs, return_tensors='pt', padding="max_length", truncation=True, max_length=24).to(self.device)

        population = initial_t['input_ids'][:, 2:-2].cpu().numpy()
        sol_per_pop = len(initial_t['input_ids'])
        num_weights = population.shape[1]
        if self.pop_size == (None, None):
            self.pop_size = (sol_per_pop, num_weights)

        for generation in tqdm(range(self.num_generations), desc="Running genetic algorithm"):
            time.sleep(0.05)
            index = self.filter_func(population)
            population_filter = []
            parents = []
            for i, token in enumerate(population):
                if i in index:
                    population_filter.append(token)
                else:
                    parents.append(token)
            
            population_filter = np.array(population_filter)
            parents = np.array(parents)
            if population_filter.size > 0:
                for row in population_filter:
                    if not np.any(np.all(self.pp_in == row, axis=1)):
                        self.pp_in = np.vstack((self.pp_in, row))
            
            # Generating next generation using crossover.
            if population_filter.size > 0:
                offspring_crossover = self.crossover(population_filter, offspring_size=(self.pop_size[0] - parents.shape[0], num_weights))
            else:
                offspring_crossover = self.crossover(parents, offspring_size=(self.pop_size[0] - parents.shape[0], num_weights))

            # Adding some variations to the offspring using mutation.
            offspring_mutation = self.mutation(offspring_crossover)

            # Creating the new population based on the parents and offspring.
            population[0:parents.shape[0], :] = parents
            population[parents.shape[0]:, :] = offspring_mutation

        return self.pp_in


def get_bceloss(
        test_output: torch.Tensor, 
        lbdb: torch.Tensor, 
        criterion: nn.Module
        ) -> List[torch.Tensor]:
    
    loss_list = []
    for number, test in enumerate(test_output):
        new_pred = torch.where(new_pred < 0.9, torch.tensor(0), torch.tensor(1))
        loss = criterion(lbdb.float(), new_pred.float())
        loss_list.append(loss)
    return loss_list


def filter_pred_contact(fasta_file, device, check_point_path) -> List[int]:
    index_list = []
    contacts = get_pred_contact(fasta_file, device, check_point_path)
    for index, contact in enumerate(contacts):
        if (contact[16][18] == 1) & (contact[13][16] == 1):  # the site is filtered by the site_filter function
            index_list.append(index)
    return index_list


def parse_arguments():
    parser = argparse.ArgumentParser(description="Design parameters.")

    parser.add_argument('-i', '--fasta_file', type=str, required=True, help="Path to the input fasta file.")
    parser.add_argument('-o', '--mutation_fasta_output', type=str, required=True, help='Path to save the mutation fasta file.')
    parser.add_argument('--num_parents_mating', type=int, default=8, help='Number of parents mating.')
    parser.add_argument('--num_generations', type=int, default=10000, help='Number of generations.')
    parser.add_argument('--num_mutations', type=int, default=3, help='Number of mutations.')
    
    # Parse initial arguments
    args = parser.parse_args()

    return args

def main():
    # initialize parameters
    args = parse_arguments()
    device = set_device()
    check_point_path = '../models/contact-based_model.pt'
    fasta_file = args.fasta_file
    mutation_fasta_output = args.mutation_fasta_output

    # load model
    model = load_lora_model(model_path, lora_path, device)
    dic_new = dict(zip(amino_acid.values(), amino_acid.keys()))

    # obtain the range of seq_ids
    seq_ids = list(dic_new.values())
    print(f"Seq IDs: min={min(seq_ids)}, max={max(seq_ids)}")

    # initialize genetic algorithm
    ga = GeneticAlgorithm(
        model=model,
        tokenizer=tokenizer,
        device=device,
        seq_ids=seq_ids,
        pop_size=(None, None),
        num_parents_mating=args.num_parents_mating,
        num_generations=args.num_generations,
        num_mutations=args.num_mutations
    )

    # run genetic algorithm
    final_population = ga.run(fasta_file)

    # remove special tokens and Cys token
    final_population = final_population.copy()
    final_population = np.delete(final_population, slice(0, 11), axis=0)
    new_population = []
    for i in final_population:
        temp = np.insert(i, 0, 23)
        temp = np.insert(temp, len(temp), 23)
        new_population.append(temp)
    new_population = np.array(new_population)

    # save the mutation fasta file
    with open(mutation_fasta_output, 'w') as f1:
        for i, token in enumerate(new_population):
            token = token.astype(int)
            seq = ''.join([amino_acid[j] for j in token])
            f1.write(f'>{i}\n{seq}\n')
    
    criterion = nn.BCELoss()

    lbdb_seq = extract_single_letter_sequence('../database/lbdb.cif', chain_id=None)
    lbdb = get_contacts_predictions(check_point_path, lbdb_seq, device)
    mutation_dict = get_fasta_dict(mutation_fasta_output)
    mutation_outputs = get_contacts_predictions(check_point_path, mutation_fasta_output, device)
    mutation_pred_contacts = get_pred_contact(mutation_fasta_output, device, check_point_path)

    # filter the mutation with plan a
    mutation_index = filter_pred_contact(mutation_fasta_output)
    pp_filter = mutation_outputs[mutation_index]
    loss_list = get_bceloss(pp_filter, lbdb, criterion)
    bceloss = torch.tensor(loss_list)
    values, indices = torch.topk(bceloss, k=5, largest=False)
    top_sequences_plan_a = []
    for i in indices:
        seq = mutation_dict[list(mutation_dict.keys())[mutation_index[i]]]
        top_sequences_plan_a.append(seq)
    print("Plana top 5 sequence:")
    for seq in top_sequences_plan_a:
        print(seq)

    # filter the mutation with plan b

    contactmodel = load_model_checkpoint('../models/contactmap_filter_planb.pt', device)
    mutation_pred_contacts = mutation_pred_contacts.float().to(device)
    contact_outputs = contactmodel(mutation_pred_contacts)
    _, contact_prediction = contact_outputs.max(dim=1)
    index = np.where(contact_prediction.cpu() == 1)
    pp_filter = mutation_outputs[index[0]]
    loss_list_plan_b = get_bceloss(pp_filter, lbdb, criterion)
    bceloss_plan_b = torch.tensor(loss_list_plan_b)
    values_b, indices_b = torch.topk(bceloss_plan_b, k=5, largest=False)
    top_sequences_plan_b = []
    for i in indices_b:
        seq = mutation_dict[list(mutation_dict.keys())[index[0][i]]]
        top_sequences_plan_b.append(seq)
    print("Plana top 5 sequence:")
    for seq in top_sequences_plan_b:
        print(seq)


if __name__ == "__main__":
    main()
