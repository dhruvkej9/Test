from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import os

app = FastAPI()

MAX_ATOMS = 50

class MoleculeRequest(BaseModel):
    smiles: str

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
    feats = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
    feats = feats[:max_atoms] + [[0]]*(max_atoms-len(feats))
    return torch.tensor(feats, dtype=torch.float)

def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > max_atoms:
        return None, None
    x = atom_features(mol, max_atoms)
    adj = rdmolops.GetAdjacencyMatrix(mol)
    edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
    return x, edge_index

# Initialize model
ATOM_FEATURES = 1
model = GNN(num_node_features=ATOM_FEATURES, hidden_channels=16, num_outputs=2)
model.eval()

# Try to load trained model if available
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "backend", "gnn_trained.pth")
if os.path.exists(MODEL_PATH):
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
    except Exception:
        pass  # Use default initialized model if loading fails

@app.post("/")
def predict_property(request: MoleculeRequest):
    try:
        x, edge_index = smiles_to_graph(request.smiles)
        if x is None or edge_index is None:
            return {
                "error": f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms", 
                "toxicity": None, 
                "solubility": None, 
                "intoxicant": None, 
                "prediction": None
            }
        
        with torch.no_grad():
            pred = model(x, edge_index)
        
        tox = float(pred[0].item()) if pred.shape[0] > 0 else None
        sol = float(pred[1].item()) if pred.shape[0] > 1 else None
        intoxicant = 1 if tox and tox > 0.5 else 0
        
        return {
            "toxicity": tox,
            "solubility": sol,
            "intoxicant": intoxicant,
            "prediction": tox
        }
    except Exception as e:
        return {
            "error": str(e), 
            "toxicity": None, 
            "solubility": None, 
            "intoxicant": None, 
            "prediction": None
        }

# For Vercel serverless compatibility
def handler(request):
    return app(request)