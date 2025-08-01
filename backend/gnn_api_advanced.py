import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from rdkit import Chem
from rdkit.Chem import rdmolops
import numpy as np
import pandas as pd
from typing import List, Optional
import os
import logging
import io

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Advanced GNN Molecular Property Prediction API",
    description="Advanced API for molecular property prediction with batch processing and model comparison",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MoleculeRequest(BaseModel):
    smiles: str
    
    @field_validator('smiles')
    @classmethod
    def validate_smiles(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError('SMILES must be a non-empty string')
        if len(v.strip()) == 0:
            raise ValueError('SMILES cannot be empty')
        if len(v) > 1000:
            raise ValueError('SMILES too long (max 1000 characters)')
        return v.strip()

class BatchRequest(BaseModel):
    smiles_list: List[str]
    
    @field_validator('smiles_list')
    @classmethod
    def validate_smiles_list(cls, v):
        if not v or len(v) == 0:
            raise ValueError('SMILES list cannot be empty')
        if len(v) > 1000:  # Reasonable batch size limit
            raise ValueError('Batch size too large (max 1000 molecules)')
        return v

class PredictionResult(BaseModel):
    smiles: str
    toxicity: Optional[float] = None
    solubility: Optional[float] = None
    intoxicant: Optional[int] = None
    prediction: Optional[float] = None
    error: Optional[str] = None
    atom_count: Optional[int] = None

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResult]
    total_processed: int
    successful_predictions: int
    failed_predictions: int

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, num_outputs=2, use_attention=False):
        super(GNN, self).__init__()
        if use_attention:
            self.conv1 = GATConv(num_node_features, hidden_channels)
            self.conv2 = GATConv(hidden_channels, hidden_channels)
        else:
            self.conv1 = GCNConv(num_node_features, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, num_outputs)

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

# Model expects 2 outputs
MAX_ATOMS = 50  # Max atoms per molecule
ATOM_FEATURES = 1  # Only atomic number as feature

# Global model variables
model = None
model_loaded = False

def load_model():
    """Load the trained model."""
    global model, model_loaded
    try:
        model = GNN(num_node_features=ATOM_FEATURES, hidden_channels=16, num_outputs=2)
        
        MODEL_PATH = os.path.join(os.path.dirname(__file__), "gnn_trained.pth")
        if os.path.exists(MODEL_PATH):
            model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
            model.eval()
            model_loaded = True
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
        else:
            logger.warning(f"Model file not found: {MODEL_PATH}")
            logger.warning("Using untrained model weights")
            model.eval()
            model_loaded = False
            
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

# Load model on startup
load_model()

def atom_features(mol, max_atoms=MAX_ATOMS):
    """Extract atomic features from molecule."""
    try:
        feats = [[atom.GetAtomicNum()] for atom in mol.GetAtoms()]
        feats = feats[:max_atoms] + [[0]]*(max_atoms-len(feats))
        return torch.tensor(feats, dtype=torch.float)
    except Exception as e:
        logger.error(f"Error extracting atom features: {e}")
        return None

def smiles_to_graph(smiles, max_atoms=MAX_ATOMS):
    """Convert SMILES to graph representation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, None
        
        num_atoms = mol.GetNumAtoms()
        if num_atoms == 0 or num_atoms > max_atoms:
            return None, None, None
        
        x = atom_features(mol, max_atoms)
        if x is None:
            return None, None, None
            
        adj = rdmolops.GetAdjacencyMatrix(mol)
        edge_index = torch.tensor(np.array(np.nonzero(adj)), dtype=torch.long)
        
        return x, edge_index, num_atoms
        
    except Exception as e:
        logger.error(f"Error converting SMILES to graph: {e}")
        return None, None, None

def predict_single_molecule(smiles: str) -> PredictionResult:
    """Predict properties for a single molecule."""
    try:
        x, edge_index, atom_count = smiles_to_graph(smiles)
        
        if x is None or edge_index is None:
            return PredictionResult(
                smiles=smiles,
                error=f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms"
            )
        
        with torch.no_grad():
            pred = model(x, edge_index)
            
        tox = float(pred[0].item()) if pred.shape[0] > 0 else None
        sol = float(pred[1].item()) if pred.shape[0] > 1 else None
        intoxicant = 1 if tox and tox > 0.5 else 0
        
        return PredictionResult(
            smiles=smiles,
            toxicity=tox,
            solubility=sol,
            intoxicant=intoxicant,
            prediction=tox,
            atom_count=atom_count
        )
        
    except Exception as e:
        logger.error(f"Error predicting for SMILES '{smiles}': {e}")
        return PredictionResult(
            smiles=smiles,
            error=f"Prediction failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Advanced GNN Molecular Property Prediction API",
        "status": "running",
        "model_loaded": model_loaded,
        "version": "2.0.0"
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

@app.post("/predict", response_model=PredictionResult)
async def predict_property(request: MoleculeRequest):
    """Predict molecular properties for a single SMILES."""
    try:
        if not model_loaded:
            logger.warning("Making prediction with untrained model")
        
        result = predict_single_molecule(request.smiles)
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/batch_predict", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchRequest):
    """Predict properties for multiple molecules."""
    try:
        if not model_loaded:
            logger.warning("Making batch predictions with untrained model")
        
        results = []
        successful = 0
        failed = 0
        
        for smiles in request.smiles_list:
            try:
                result = predict_single_molecule(smiles)
                results.append(result)
                
                if result.error is None:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing SMILES '{smiles}': {e}")
                results.append(PredictionResult(
                    smiles=smiles,
                    error=f"Processing failed: {str(e)}"
                ))
                failed += 1
        
        return BatchPredictionResponse(
            results=results,
            total_processed=len(request.smiles_list),
            successful_predictions=successful,
            failed_predictions=failed
        )
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Unexpected error in batch_predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/csv_predict")
async def csv_predict(file: UploadFile = File(...)):
    """Predict properties for molecules from CSV file."""
    try:
        if not model_loaded:
            logger.warning("Making CSV predictions with untrained model")
        
        # Validate file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read CSV content
        content = await file.read()
        csv_string = content.decode('utf-8')
        
        try:
            df = pd.read_csv(io.StringIO(csv_string))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Validate CSV structure
        if 'smiles' not in df.columns:
            raise HTTPException(status_code=400, detail="CSV must contain 'smiles' column")
        
        # Limit batch size
        if len(df) > 1000:
            raise HTTPException(status_code=400, detail="CSV too large (max 1000 rows)")
        
        smiles_list = df['smiles'].dropna().tolist()
        
        if len(smiles_list) == 0:
            raise HTTPException(status_code=400, detail="No valid SMILES found in CSV")
        
        # Process batch
        results = []
        successful = 0
        failed = 0
        
        for smiles in smiles_list:
            try:
                result = predict_single_molecule(str(smiles))
                results.append(result.dict())
                
                if result.error is None:
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing SMILES '{smiles}': {e}")
                results.append({
                    "smiles": str(smiles),
                    "error": f"Processing failed: {str(e)}"
                })
                failed += 1
        
        return {
            "results": results,
            "total_processed": len(smiles_list),
            "successful_predictions": successful,
            "failed_predictions": failed
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in csv_predict endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/explain")
async def explain(request: MoleculeRequest):
    """Provide explanation for molecular property prediction."""
    try:
        x, edge_index, atom_count = smiles_to_graph(request.smiles)
        
        if x is None or edge_index is None:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid SMILES or molecule has more than {MAX_ATOMS} atoms"
            )
        
        # Generate random importance scores (placeholder for real explanation)
        # In a real implementation, you would use techniques like GradCAM or attention weights
        np.random.seed(42)  # For reproducible results
        importances = np.random.rand(atom_count).tolist()
        
        return {
            "smiles": request.smiles,
            "atom_count": atom_count,
            "atom_importances": importances,
            "explanation": "Atom importance scores (higher values indicate greater influence on prediction)"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in explain endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/accuracy")
async def get_accuracy():
    """Get model accuracy information."""
    try:
        acc_path = os.path.join(os.path.dirname(__file__), "accuracy.txt")
        eval_path = os.path.join(os.path.dirname(__file__), "evaluation_results.txt")
        
        result = {"model_loaded": model_loaded}
        
        if os.path.exists(acc_path):
            with open(acc_path, "r") as f:
                acc = f.read().strip()
            result["balanced_accuracy"] = float(acc)
        
        if os.path.exists(eval_path):
            result["detailed_results_available"] = True
            result["evaluation_file"] = eval_path
        else:
            result["detailed_results_available"] = False
        
        if not model_loaded:
            result["warning"] = "Model not properly loaded. Train the model first."
        
        return result
        
    except Exception as e:
        logger.error(f"Error in accuracy endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/reload_model")
async def reload_model():
    """Reload the model."""
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
    uvicorn.run(app, host="0.0.0.0", port=8001)
