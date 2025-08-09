# Drug Toxicity Prediction using Graph Neural Networks

A machine learning application that predicts drug and molecule toxicity using Graph Neural Networks (GNN). The system features an interactive Streamlit web interface and a FastAPI backend for both batch predictions and individual molecule analysis.

## 🚀 Features

- **Interactive Web Interface**: User-friendly Streamlit application for drug toxicity prediction
- **Batch Processing**: Upload CSV files for bulk molecule analysis
- **Real-time Predictions**: Enter SMILES strings or common drug names for instant predictions
- **3D Molecular Visualization**: Interactive 3D molecule viewer powered by py3Dmol
- **RESTful API**: FastAPI backend for programmatic access
- **Graph Neural Networks**: Advanced GNN models using PyTorch Geometric
- **Comprehensive Dataset**: Built on the Tox21 dataset for robust predictions

## 🏗️ Architecture

```
├── frontend/           # Streamlit web application
│   ├── app.py         # Main application entry point
│   ├── app_basic.py   # Simplified version
│   ├── pages/         # Multi-page application structure
│   └── src/           # Components and utilities
├── backend/           # FastAPI backend services
│   ├── gnn_api.py     # Main API endpoints
│   ├── train_gnn.py   # Model training scripts
│   ├── eval_gnn.py    # Model evaluation
│   └── agent.py       # Additional API routes
├── data/              # Dataset and preprocessing
│   ├── download_tox21.py     # Dataset download script
│   ├── preprocess_tox21.py   # Data preprocessing
│   └── tox21_*.csv          # Tox21 dataset files
└── requirements.txt   # Python dependencies
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dhruvkej9/Test.git
   cd Test
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess the dataset** (if needed)
   ```bash
   cd data
   python download_tox21.py
   python preprocess_tox21.py
   cd ..
   ```

4. **Train the model** (optional - pre-trained model included)
   ```bash
   cd backend
   python train_gnn.py
   cd ..
   ```

## 🚀 Usage

### Web Interface (Streamlit)

Start the interactive web application:

```bash
cd frontend
streamlit run app.py
```

The application will be available at `http://localhost:8501`

**Features available in the web interface:**
- **Single Molecule Prediction**: Enter SMILES strings or common drug names
- **Batch Prediction**: Upload CSV files with SMILES data
- **3D Visualization**: Interactive molecular structure viewer
- **Toxicity Analysis**: Detailed toxicity predictions with confidence scores

### API Backend (FastAPI)

Start the API server:

```bash
cd backend
uvicorn gnn_api:app --reload
```

The API will be available at `http://localhost:8000`

**API Endpoints:**

- `POST /predict`: Predict toxicity for a single molecule
- `GET /docs`: Interactive API documentation
- `GET /health`: Health check endpoint

**Example API usage:**

```python
import requests

# Predict toxicity for aspirin
response = requests.post(
    "http://localhost:8000/predict",
    json={"smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"}
)
result = response.json()
print(f"Toxicity prediction: {result}")
```

## 📊 Model Performance

The Graph Neural Network model is trained on the Tox21 dataset and achieves:
- Training accuracy: ~85% (see `backend/accuracy.txt`)
- Validation performance on multiple toxicity endpoints
- Support for molecules up to 50 atoms

## 🧪 Supported Molecules

The system supports predictions for various types of molecules:

- **Common drugs**: Aspirin, Paracetamol, Ibuprofen, Caffeine, etc.
- **Chemical compounds**: Benzene, Acetone, Ethanol, etc.
- **Custom molecules**: Any valid SMILES string

## 📝 Input Formats

### SMILES Strings
```
CC(=O)OC1=CC=CC=C1C(=O)O  # Aspirin
CCO                        # Ethanol
CN1C=NC2=C1C(=O)N(C(=O)N2C)C  # Caffeine
```

### CSV Format for Batch Processing
```csv
smiles,name
CC(=O)OC1=CC=CC=C1C(=O)O,aspirin
CCO,ethanol
CN1C=NC2=C1C(=O)N(C(=O)N2C)C,caffeine
```

## 🔬 Technology Stack

- **Machine Learning**: PyTorch, PyTorch Geometric
- **Web Framework**: Streamlit, FastAPI
- **Chemistry**: RDKit, py3Dmol
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, stmol
- **API**: Uvicorn, Pydantic

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 📚 References

- [Tox21 Challenge Dataset](https://tripod.nih.gov/tox21/challenge/)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
- [RDKit Documentation](https://www.rdkit.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed with `pip install -r requirements.txt`
2. **Model not found**: Run the training script or ensure the pre-trained model exists
3. **Port conflicts**: Change the port in the startup commands if default ports are occupied

### Getting Help

- Check the [Issues](../../issues) page for known problems
- Review the API documentation at `/docs` when running the FastAPI server
- Ensure all dependencies are correctly installed

## 🎯 Future Enhancements

- [ ] Support for larger molecules (>50 atoms)
- [ ] Additional toxicity endpoints
- [ ] Improved model architectures
- [ ] Docker containerization
- [ ] Batch API endpoints
- [ ] Model explainability features

---

**Built with ❤️ for drug discovery and computational chemistry**