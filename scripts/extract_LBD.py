#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
extract_LBD.py

This script extracts LBD sequences from a FASTA-formatted input file and saves the results to an output file.

Usage:
    python extract_LBD.py -i input_file_path -o output_file_path

Example:
    python extract_LBD.py -i ../database/ALF.fasta -o ../outputs/ALF_LBD.fasta
"""

import argparse
import os
from utils import parse_fasta

def find_residue_distance(sequence, target_residue, distance):
    """
    Finds the LBD region in a sequence where two target residues are separated by a specified distance.

    sequence: Protein sequence
    target_residue: Target residue (e.g., 'C')
    distance: The distance between two target residues
    """
    positions = [i for i, res in enumerate(sequence) if res == target_residue]
    LBD = ''
    if len(positions) > 1:
        for i in range(len(positions) - 1):
            if positions[i + 1] - positions[i] == distance + 1:
                LBD = sequence[positions[i]:positions[i + 1] + 1]
                break  # Remove this line if you want to find all matching LBD regions
    return LBD

def extract_LBD(input_file, output_file, target_residue='C', distance=20):
    """
    Extracts LBD sequences from the input FASTA file and writes them to the output file.
    """
    # Parse the input FASTA file
    sequences = parse_fasta(input_file)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write LBD sequences to the output file
    with open(output_file, 'w') as f_out:
        for header, sequence in sequences.items():
            LBD = find_residue_distance(sequence, target_residue, distance)
            if LBD:
                f_out.write(f'>{header}\n{LBD}\n')

    print(f"LBD sequences have been successfully extracted and saved to {output_file}")

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description='Extract LBD sequences from a FASTA file')
    parser.add_argument('-i', '--input', required=True, help='Path to the input FASTA file')
    parser.add_argument('-o', '--output', required=True, help='Path to the output FASTA file')
    parser.add_argument('-r', '--residue', default='C', help='Target residue (default: C)')
    parser.add_argument('-d', '--distance', type=int, default=20, help='Distance between target residues (default: 20)')

    args = parser.parse_args()

    # Call the extraction function
    extract_LBD(args.input, args.output, args.residue, args.distance)

if __name__ == '__main__':
    main()