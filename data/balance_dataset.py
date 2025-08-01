#!/usr/bin/env python3
"""
Dataset balancing utility for handling imbalanced molecular datasets.
Supports class weight calculation and simple oversampling.
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
from rdkit import Chem
import os
import argparse


def calculate_class_weights(labels):
    """
    Calculate class weights for imbalanced dataset.
    Returns a dictionary mapping class labels to weights.
    """
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=unique_labels,
        y=labels
    )
    
    weight_dict = dict(zip(unique_labels, class_weights))
    return weight_dict


def oversample_minority_class(df, smiles_col='smiles', label_col='label', random_state=42):
    """
    Simple oversampling by duplicating minority class samples.
    Returns a new balanced DataFrame.
    """
    np.random.seed(random_state)
    
    # Get class counts
    class_counts = df[label_col].value_counts()
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()
    
    majority_count = class_counts[majority_class]
    minority_count = class_counts[minority_class]
    
    print(f"Original class distribution:")
    print(f"  Class {majority_class}: {majority_count} samples")
    print(f"  Class {minority_class}: {minority_count} samples")
    
    # Separate majority and minority samples
    majority_samples = df[df[label_col] == majority_class]
    minority_samples = df[df[label_col] == minority_class]
    
    # Calculate how many minority samples to add
    samples_to_add = majority_count - minority_count
    
    # Randomly sample from minority class with replacement
    additional_samples = minority_samples.sample(n=samples_to_add, replace=True, random_state=random_state)
    
    # Combine all samples
    balanced_df = pd.concat([majority_samples, minority_samples, additional_samples], ignore_index=True)
    
    # Shuffle the balanced dataset
    balanced_df = balanced_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    print(f"Balanced class distribution:")
    print(balanced_df[label_col].value_counts())
    
    return balanced_df


def save_class_weights(class_weights, output_path):
    """
    Save class weights to a file for use in training.
    """
    with open(output_path, 'w') as f:
        f.write("# Class weights for imbalanced dataset training\n")
        f.write("# Format: class_label: weight\n")
        for label, weight in class_weights.items():
            f.write(f"{label}: {weight:.6f}\n")
    
    print(f"Class weights saved to: {output_path}")


def load_class_weights(weights_path):
    """
    Load class weights from a file.
    Returns a dictionary mapping class labels to weights.
    """
    weights = {}
    with open(weights_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split(': ')
            if len(parts) == 2:
                label = int(parts[0])
                weight = float(parts[1])
                weights[label] = weight
    return weights


def main():
    parser = argparse.ArgumentParser(description="Balance molecular dataset for training")
    parser.add_argument("--input", "-i", required=True, help="Input CSV file path")
    parser.add_argument("--output", "-o", help="Output CSV file path (for oversampling)")
    parser.add_argument("--method", "-m", choices=['oversample', 'weights', 'both'], 
                       default='both', help="Balancing method")
    parser.add_argument("--smiles-col", default='smiles', help="SMILES column name")
    parser.add_argument("--label-col", default='label', help="Label column name")
    parser.add_argument("--weights-output", help="Output file for class weights")
    
    args = parser.parse_args()
    
    # Load dataset
    print(f"Loading dataset from: {args.input}")
    df = pd.read_csv(args.input)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:")
    print(df[args.label_col].value_counts())
    print(f"Class percentages:")
    print(df[args.label_col].value_counts(normalize=True) * 100)
    
    # Calculate class weights
    if args.method in ['weights', 'both']:
        class_weights = calculate_class_weights(df[args.label_col].values)
        print(f"\nCalculated class weights: {class_weights}")
        
        weights_output = args.weights_output or args.input.replace('.csv', '_class_weights.txt')
        save_class_weights(class_weights, weights_output)
    
    # Apply oversampling
    if args.method in ['oversample', 'both']:
        balanced_df = oversample_minority_class(df, args.smiles_col, args.label_col)
        
        output_path = args.output or args.input.replace('.csv', '_balanced.csv')
        balanced_df.to_csv(output_path, index=False)
        print(f"\nBalanced dataset saved to: {output_path}")


if __name__ == "__main__":
    main()