import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import os
from sklearn.metrics import (
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score
)
from collections import Counter

MAX_ATOMS = 60

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=2):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_outputs)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return torch.sigmoid(x.mean(dim=0))

def atom_features(mol, max_atoms=MAX_ATOMS):
    feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [0]*(max_atoms-len(feats))
    return torch.tensor([feats], dtype=torch.float).T

def load_dataset(csv_path, max_atoms=MAX_ATOMS):
    df = pd.read_csv(csv_path)
    data_list = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['label']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if mol.GetNumAtoms() > max_atoms:
            continue
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        y = torch.tensor([label, 1-label], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def evaluate_model_comprehensive(model, val_data, threshold=0.5):
    """
    Comprehensive evaluation of model performance on imbalanced dataset.
    Returns detailed metrics including precision, recall, F1, AUC, etc.
    """
    if not val_data:
        print("No validation data available.")
        return None
    
    model.eval()
    val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
    
    predictions = []
    prediction_probs = []
    true_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            out = model(batch.x, batch.edge_index)
            prob_tox = out[0].item()  # Probability of toxicity
            pred_tox = (prob_tox > threshold)
            true_tox = int(batch.y[0].item())
            
            predictions.append(int(pred_tox))
            prediction_probs.append(prob_tox)
            true_labels.append(true_tox)
    
    # Convert to numpy arrays
    y_true = np.array(true_labels)
    y_pred = np.array(predictions)
    y_prob = np.array(prediction_probs)
    
    # Class distribution
    class_dist = Counter(true_labels)
    print(f"Validation set class distribution: {class_dist}")
    
    # Basic accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Balanced accuracy (handles class imbalance better)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-score for each class
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # Macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # AUC-ROC and AUC-PR (important for imbalanced datasets)
    try:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr = average_precision_score(y_true, y_prob)
    except ValueError:
        # Handle case where only one class is present
        auc_roc = 0.5
        auc_pr = np.mean(y_true)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate specificity and sensitivity manually
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0  # Recall for positive class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0  # Recall for negative class
    
    # Create results dictionary
    results = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'precision_class_0': precision[0] if len(precision) > 0 else 0,
        'recall_class_0': recall[0] if len(recall) > 0 else 0,
        'f1_class_0': f1[0] if len(f1) > 0 else 0,
        'precision_class_1': precision[1] if len(precision) > 1 else 0,
        'recall_class_1': recall[1] if len(recall) > 1 else 0,
        'f1_class_1': f1[1] if len(f1) > 1 else 0,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm,
        'support_class_0': support[0] if len(support) > 0 else 0,
        'support_class_1': support[1] if len(support) > 1 else 0
    }
    
    return results

def print_evaluation_results(results):
    """Print comprehensive evaluation results in a readable format."""
    print("\n" + "="*60)
    print("COMPREHENSIVE MODEL EVALUATION RESULTS")
    print("="*60)
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Accuracy:          {results['accuracy']:.3f}")
    print(f"  Balanced Accuracy: {results['balanced_accuracy']:.3f}")
    print(f"  AUC-ROC:          {results['auc_roc']:.3f}")
    print(f"  AUC-PR:           {results['auc_pr']:.3f}")
    
    print(f"\nCLASS-WISE PERFORMANCE:")
    print(f"  Class 0 (Non-toxic):")
    print(f"    Precision: {results['precision_class_0']:.3f}")
    print(f"    Recall:    {results['recall_class_0']:.3f}")
    print(f"    F1-Score:  {results['f1_class_0']:.3f}")
    print(f"    Support:   {results['support_class_0']}")
    
    print(f"  Class 1 (Toxic):")
    print(f"    Precision: {results['precision_class_1']:.3f}")
    print(f"    Recall:    {results['recall_class_1']:.3f}")
    print(f"    F1-Score:  {results['f1_class_1']:.3f}")
    print(f"    Support:   {results['support_class_1']}")
    
    print(f"\nAVERAGED METRICS:")
    print(f"  Macro Avg:")
    print(f"    Precision: {results['macro_precision']:.3f}")
    print(f"    Recall:    {results['macro_recall']:.3f}")
    print(f"    F1-Score:  {results['macro_f1']:.3f}")
    
    print(f"  Weighted Avg:")
    print(f"    Precision: {results['weighted_precision']:.3f}")
    print(f"    Recall:    {results['weighted_recall']:.3f}")
    print(f"    F1-Score:  {results['weighted_f1']:.3f}")
    
    print(f"\nSPECIFIC METRICS (for imbalanced datasets):")
    print(f"  Sensitivity (True Positive Rate):  {results['sensitivity']:.3f}")
    print(f"  Specificity (True Negative Rate):  {results['specificity']:.3f}")
    
    print(f"\nCONFUSION MATRIX:")
    print(f"  {results['confusion_matrix']}")
    print("  [TN FP]")
    print("  [FN TP]")
    
    print("="*60)

def save_evaluation_results(results, output_path):
    """Save evaluation results to a file."""
    with open(output_path, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"Overall Performance:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Balanced Accuracy: {results['balanced_accuracy']:.4f}\n")
        f.write(f"AUC-ROC: {results['auc_roc']:.4f}\n")
        f.write(f"AUC-PR: {results['auc_pr']:.4f}\n\n")
        
        f.write(f"Class-wise Performance:\n")
        f.write(f"Class 0 - Precision: {results['precision_class_0']:.4f}, Recall: {results['recall_class_0']:.4f}, F1: {results['f1_class_0']:.4f}\n")
        f.write(f"Class 1 - Precision: {results['precision_class_1']:.4f}, Recall: {results['recall_class_1']:.4f}, F1: {results['f1_class_1']:.4f}\n\n")
        
        f.write(f"Averaged Metrics:\n")
        f.write(f"Macro F1: {results['macro_f1']:.4f}\n")
        f.write(f"Weighted F1: {results['weighted_f1']:.4f}\n\n")
        
        f.write(f"Imbalanced Dataset Metrics:\n")
        f.write(f"Sensitivity: {results['sensitivity']:.4f}\n")
        f.write(f"Specificity: {results['specificity']:.4f}\n\n")
        
        f.write(f"Confusion Matrix:\n{results['confusion_matrix']}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate GNN model on imbalanced dataset")
    parser.add_argument("--dataset", default=None, help="Path to evaluation dataset (default: use validation split)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    parser.add_argument("--model-path", default=None, help="Path to trained model")
    
    args = parser.parse_args()
    
    # Load dataset
    if args.dataset:
        data_list = load_dataset(args.dataset)
        val_data = data_list
    else:
        data_list = load_dataset(os.path.join("data", "tox21_sr-p53.csv"))
        split_idx = int(0.8 * len(data_list))
        val_data = data_list[split_idx:]
    
    print(f"Evaluation set size: {len(val_data)}")
    
    # Load model
    model = GNN(num_node_features=1, hidden_channels=16, num_outputs=2)
    model_path = args.model_path or os.path.join("backend", "gnn_trained.pth")
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        print(f"Model loaded from: {model_path}")
    else:
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_gnn.py")
        exit(1)
    
    # Evaluate model
    results = evaluate_model_comprehensive(model, val_data, threshold=args.threshold)
    
    if results:
        # Print results
        print_evaluation_results(results)
        
        # Save results
        output_path = os.path.join(os.path.dirname(model_path), "evaluation_results.txt")
        save_evaluation_results(results, output_path)
        print(f"\nDetailed results saved to: {output_path}")
        
        # Update accuracy.txt with balanced accuracy for better representation
        accuracy_path = os.path.join(os.path.dirname(model_path), "accuracy.txt")
        with open(accuracy_path, "w") as f:
            f.write(f"{results['balanced_accuracy']:.4f}")
        print(f"Balanced accuracy saved to: {accuracy_path}")
    else:
        print("No evaluation data available.")