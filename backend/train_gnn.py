import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from rdkit import Chem
from rdkit.Chem import rdmolops
import pandas as pd
import numpy as np
import os
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
from collections import Counter

def atom_features(mol, max_atoms=10):
    # Simple feature: atomic number, padded/truncated to max_atoms
    feats = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [0]*(max_atoms-len(feats))
    return torch.tensor([feats], dtype=torch.float).T  # shape (max_atoms, 1)

def load_dataset(csv_path, max_atoms=60):
    df = pd.read_csv(csv_path)
    data_list = []
    for idx, row in df.iterrows():
        smiles = row['smiles']
        label = row['label']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        if mol.GetNumAtoms() > max_atoms:
            continue  # Skip molecules that are too large
        x = atom_features(mol, max_atoms)
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        # For demo, use label as toxicity and (1-label) as dummy solubility
        y = torch.tensor([label, 1-label], dtype=torch.float)
        data = Data(x=x, edge_index=edge_index, y=y)
        data_list.append(data)
    return data_list

def load_class_weights(weights_path):
    """Load class weights from file."""
    if not os.path.exists(weights_path):
        return None
    
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

def train_gnn(data_list, epochs=20, batch_size=32, use_class_weights=True, balanced_dataset_path=None):
    # Use balanced dataset if provided
    if balanced_dataset_path and os.path.exists(balanced_dataset_path):
        print(f"Using balanced dataset: {balanced_dataset_path}")
        data_list = load_dataset(balanced_dataset_path)
    
    # Shuffle the data for a realistic split
    import random
    random.shuffle(data_list)
    split_idx = int(0.8 * len(data_list))
    train_data = data_list[:split_idx]
    val_data = data_list[split_idx:]
    
    # Check class distribution
    train_labels = [int(data.y[0].item()) for data in train_data]
    val_labels = [int(data.y[0].item()) for data in val_data]
    
    print(f"Training set class distribution: {Counter(train_labels)}")
    print(f"Validation set class distribution: {Counter(val_labels)}")
    
    model = GNN(num_node_features=1, hidden_channels=16, num_outputs=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Setup class weights for loss function
    class_weights_tensor = None
    if use_class_weights:
        weights_path = os.path.join(os.path.dirname(__file__), "..", "data", "tox21_sr-p53_class_weights.txt")
        class_weights = load_class_weights(weights_path)
        if class_weights:
            # Convert to tensor for use in loss function
            weight_list = [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]
            class_weights_tensor = torch.tensor(weight_list, dtype=torch.float)
            print(f"Using class weights: {class_weights}")
        else:
            print("No class weights file found, using equal weights")
    
    loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            try:
                y_reshaped = batch.y.float().view(-1, 2)
                target = y_reshaped.mean(dim=0)
                
                # Use weighted loss if class weights are available
                if class_weights_tensor is not None:
                    # Calculate weighted binary cross entropy
                    loss = F.binary_cross_entropy(out, target, weight=class_weights_tensor)
                else:
                    loss = F.binary_cross_entropy(out, target)
                    
            except Exception as e:
                print(f"batch.y shape: {batch.y.shape}, out shape: {out.shape}")
                raise e
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")
    
    # Save in backend/gnn_trained.pth
    save_path = os.path.join(os.path.dirname(__file__), "gnn_trained.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved as {save_path}")
    
    # Comprehensive evaluation on validation set
    if val_data:
        model.eval()
        val_loader = DataLoader(val_data, batch_size=1, shuffle=False)
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index)
                pred_tox = (out[0] > 0.5).float().item()
                true_tox = batch.y[0].item()
                predictions.append(int(pred_tox))
                true_labels.append(int(true_tox))
        
        # Calculate comprehensive metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, predictions, average=None, zero_division=0
        )
        
        # Calculate overall accuracy
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        acc = correct / len(true_labels) if len(true_labels) > 0 else 0.0
        
        print(f"\nValidation Results:")
        print(f"Overall Accuracy: {acc:.2%}")
        print(f"Class-wise metrics:")
        for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
            print(f"  Class {i}: Precision={p:.3f}, Recall={r:.3f}, F1={f:.3f}, Support={s}")
        
        # Macro and weighted averages
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='macro', zero_division=0
        )
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted', zero_division=0
        )
        
        print(f"Macro avg: Precision={macro_precision:.3f}, Recall={macro_recall:.3f}, F1={macro_f1:.3f}")
        print(f"Weighted avg: Precision={weighted_precision:.3f}, Recall={weighted_recall:.3f}, F1={weighted_f1:.3f}")
        
        # Confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        print(f"Confusion Matrix:")
        print(cm)
        
        # Save detailed results
        with open(os.path.join(os.path.dirname(__file__), "accuracy.txt"), "w") as f:
            f.write(f"{acc:.4f}")
        
        with open(os.path.join(os.path.dirname(__file__), "validation_results.txt"), "w") as f:
            f.write(f"Overall Accuracy: {acc:.4f}\n")
            f.write(f"Macro F1: {macro_f1:.4f}\n")
            f.write(f"Weighted F1: {weighted_f1:.4f}\n")
            f.write(f"Class 0 - Precision: {precision[0]:.4f}, Recall: {recall[0]:.4f}, F1: {f1[0]:.4f}\n")
            f.write(f"Class 1 - Precision: {precision[1]:.4f}, Recall: {recall[1]:.4f}, F1: {f1[1]:.4f}\n")
            f.write(f"Confusion Matrix:\n{cm}\n")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GNN model")
    parser.add_argument("--use-balanced", action="store_true", help="Use balanced dataset")
    parser.add_argument("--no-class-weights", action="store_true", help="Disable class weights")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Generate balanced dataset if requested
    balanced_path = None
    if args.use_balanced:
        balanced_path = os.path.join(os.path.dirname(__file__), "..", "data", "tox21_sr-p53_balanced.csv")
        if not os.path.exists(balanced_path):
            print("Generating balanced dataset...")
            import subprocess
            subprocess.run([
                "python", 
                os.path.join(os.path.dirname(__file__), "..", "data", "balance_dataset.py"),
                "--input", os.path.join(os.path.dirname(__file__), "..", "data", "tox21_sr-p53.csv"),
                "--method", "oversample"
            ])
    
    data_list = load_dataset(os.path.join(os.path.dirname(__file__), "..", "data", "tox21_sr-p53.csv"))
    train_gnn(
        data_list, 
        epochs=args.epochs, 
        batch_size=32, 
        use_class_weights=not args.no_class_weights,
        balanced_dataset_path=balanced_path
    )
