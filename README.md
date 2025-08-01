# Drug Discovery with Graph Neural Networks (GNNs)

A molecular property prediction application using Graph Neural Networks to predict toxicity, solubility, and intoxicant properties of chemical compounds. The app features both a simple and advanced interface with agentic AI capabilities.

## 🚀 Quick Deployment

This application is designed for easy, one-click deployment to Vercel or Streamlit Community Cloud with minimal setup required.

### Deploy to Vercel

1. **Fork this repository** to your GitHub account
2. **Import to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project" and select this repository
   - Vercel will automatically detect the configuration
3. **Deploy**: Click "Deploy" - no manual configuration needed!

The app will be available at your Vercel URL with:
- Main Streamlit app at the root URL
- API endpoints at `/api/predict`, `/api/agent`, `/api/accuracy`

### Deploy to Streamlit Community Cloud

1. **Fork this repository** to your GitHub account
2. **Deploy to Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app" and connect your GitHub account
   - Select this repository
   - Set the main file path to `frontend/app.py`
   - Click "Deploy"

For the advanced app, deploy separately with main file path: `frontend/app_advanced.py`

## 🏗️ Local Development

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Test
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the backend API** (for local development):
   ```bash
   cd backend
   uvicorn gnn_api_advanced:app --reload --port 8000
   ```

4. **Run the Streamlit frontend** (in a new terminal):
   ```bash
   cd frontend
   streamlit run app.py
   ```

   Or for the advanced version:
   ```bash
   streamlit run app_advanced.py
   ```

### Local API Testing

The serverless API functions can be tested locally:

```bash
# Test prediction API
cd api
uvicorn predict:app --reload --port 8001

# Test agent API  
uvicorn agent:app --reload --port 8002

# Test accuracy API
uvicorn accuracy:app --reload --port 8003
```

## 📁 Project Structure

```
├── api/                    # Vercel serverless functions
│   ├── predict.py         # Main prediction endpoint
│   ├── agent.py           # Agentic AI endpoint
│   └── accuracy.py        # Model accuracy endpoint
├── backend/               # Original backend (for local dev)
│   ├── gnn_api.py         # Basic FastAPI app
│   ├── gnn_api_advanced.py # Advanced FastAPI app
│   ├── agent.py           # Agent router
│   ├── train_gnn.py       # Model training
│   ├── eval_gnn.py        # Model evaluation
│   └── gnn_trained.pth    # Trained model weights
├── frontend/              # Streamlit applications
│   ├── app.py             # Main Streamlit app
│   ├── app_advanced.py    # Advanced Streamlit app
│   ├── pages/             # Multi-page Streamlit components
│   └── src/               # Shared components
├── data/                  # Data processing scripts
├── vercel.json           # Vercel deployment configuration
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## 🎯 Features

### Main Application (`app.py`)
- Simple molecular property prediction interface
- SMILES string input and validation
- 2D/3D molecular visualization
- Toxicity, solubility, and intoxicant prediction

### Advanced Application (`app_advanced.py`)
- Enhanced UI with professional styling
- Agentic AI assistant for conversational interaction
- PubChem integration for compound information
- Batch prediction capabilities
- Detailed explanations and visualizations
- Memory-based agent interactions

### API Endpoints

- **`/api/predict`**: Predict molecular properties from SMILES
- **`/api/agent`**: Agentic AI for advanced interactions
- **`/api/accuracy`**: Get model accuracy metrics

## 🧪 Usage Examples

### Simple Prediction
1. Enter a SMILES string (e.g., `CCO` for ethanol)
2. Click "Predict"
3. View results with molecular structure

### Advanced Features
1. Use compound names (e.g., "aspirin", "caffeine")
2. Interact with the AI agent in the sidebar
3. Upload CSV files for batch prediction
4. Explore PubChem compound details

### API Usage
```python
import requests

# Predict molecular properties
response = requests.post("/api/predict", json={"smiles": "CCO"})
result = response.json()

# Use agentic AI
response = requests.post("/api/agent", json={
    "goal": "predict toxicity",
    "smiles": "CCO"
})
```

## 🔧 Configuration

### Environment Variables
- `VERCEL`: Automatically set by Vercel, used to switch between local and cloud API endpoints

### Model Configuration
- `MAX_ATOMS`: Maximum number of atoms per molecule (default: 50)
- The application uses a Graph Neural Network (GNN) trained on molecular data
- Pre-trained weights are loaded from `backend/gnn_trained.pth` if available

## 🚨 Important Notes

### For Production Use
- The current model is for demonstration purposes
- For production use, train the model on your specific dataset
- Consider adding authentication and rate limiting for API endpoints
- Validate all inputs thoroughly for security

### Limitations
- Maximum molecule size: 50 atoms
- Model predictions should be validated experimentally
- PubChem integration requires internet connectivity
- Some dependencies may have large footprints for serverless deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test locally
5. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🔗 Links

- [Vercel Documentation](https://vercel.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io)
- [RDKit Documentation](https://rdkit.readthedocs.io)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)