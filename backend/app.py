from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

# Example GNN import placeholder
# from model import MyGNNModel

app = FastAPI()

# Allow CORS for Hugging Face Spaces
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Example input schema
class PredictRequest(BaseModel):
    features: list

# Example: load your model here (update with your actual model)
# model = MyGNNModel()
# model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))

@app.get("/")
def read_root():
    return {"msg": "Backend is running!"}

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        features = np.array(request.features)
        # prediction = model.predict(features)
        # Example response:
        prediction = float(np.sum(features))
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
