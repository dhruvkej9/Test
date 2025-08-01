import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import logging
import os
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="GNN Molecular Property Prediction API",
    description="API for predicting molecular properties using Graph Neural Networks",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_ATOMS = 50  # Increased from 10 to 50 for robust support

class MoleculeRequest(BaseModel):
    smiles: str
    
    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('SMILES must be a non-empty string')
        if len(v.strip()) == 0:
            raise ValueError('SMILES cannot be empty')
        if len(v) > 1000:  # Reasonable limit
            raise ValueError('SMILES too long (max 1000 characters)')
        return v.strip()

class PredictionResponse(BaseModel):
    prediction: Optional[float] = None
    error: Optional[str] = None
    smiles: Optional[str] = None
    valid_molecule: bool = True
    atom_count: Optional[int] = None

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        try:
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.lin(x)
            return torch.sigmoid(x.mean(dim=0))
        except Exception as e:
            logger.error(f"Error in model forward pass: {e}")
            raise

def smiles_to_graph(smiles: str):
    """
    Convert SMILES string to graph representation.
    Returns (features, edge_index, atom_count) or (None, None, None) if invalid.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None, None, None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0:
            logger.warning(f"Empty molecule from SMILES: {smiles}")
            return None, None, None
        
        if num_atoms > MAX_ATOMS:
            logger.warning(f"Molecule too large ({num_atoms} atoms > {MAX_ATOMS}): {smiles}")
            return None, None, None
        
        # Create node features (identity matrix approach)
        x = np.eye(MAX_ATOMS, dtype=np.float32)
        x[:num_atoms, :num_atoms] = np.eye(num_atoms, dtype=np.float32)
        
        # Get adjacency matrix and edge indices
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = np.array(np.nonzero(adj))
        
        # Convert to tensors
        x_tensor = torch.tensor(x, dtype=torch.float)
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        
        return x_tensor, edge_index_tensor, num_atoms
        
    except Exception as e:
        logger.error(f"Error processing SMILES '{smiles}': {e}")
        return None, None, None

# Initialize model
model = None
model_loaded = False

def load_model():
    """Load the trained GNN model."""
    global model, model_loaded
    try:
        model = GNN(num_node_features=MAX_ATOMS, hidden_channels=16)
        model_path = os.path.join(os.path.dirname(__file__), "gnn_trained.pth")
        
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            model_loaded = True
            logger.info(f"Model loaded successfully from {model_path}")
        else:
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Model predictions will use random weights")
            model.eval()
            model_loaded = False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

# Load model on startup
load_model()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "GNN Molecular Property Prediction API",
        "status": "running",
        "model_loaded": model_loaded
    }

@app.get("/health")
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "model_loaded": model_loaded,
        "max_atoms": MAX_ATOMS,
        "model_path_exists": os.path.exists(os.path.join(os.path.dirname(__file__), "gnn_trained.pth"))
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_property(request: MoleculeRequest):
    """
    Predict molecular properties for a given SMILES string.
    Returns prediction score between 0 and 1.
    """
    try:
        if not model_loaded:
            logger.warning("Making prediction with untrained model")
        
        # Validate and process SMILES
        smiles = request.smiles
        x, edge_index, atom_count = smiles_to_graph(smiles)
        
        if x is None or edge_index is None:
            return PredictionResponse(
                error=f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms",
                smiles=smiles,
                valid_molecule=False
            )
        
        # Make prediction
        with torch.no_grad():
            try:
                pred = model(x, edge_index)
                prediction_value = float(pred.item())
                
                # Ensure prediction is in valid range
                prediction_value = max(0.0, min(1.0, prediction_value))
                
                return PredictionResponse(
                    prediction=prediction_value,
                    smiles=smiles,
                    valid_molecule=True,
                    atom_count=atom_count
                )
                
            except Exception as e:
                logger.error(f"Error during model prediction: {e}")
                return PredictionResponse(
                    error=f"Model prediction failed: {str(e)}",
                    smiles=smiles,
                    valid_molecule=True,
                    atom_count=atom_count
                )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/reload_model")
async def reload_model():
    """Reload the model (useful for updating with newly trained models)."""
    try:
        load_model()
        return {
            "message": "Model reloaded",
            "model_loaded": model_loaded
        }
    except Exception as e:
        logger.error(f"Error reloading model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
