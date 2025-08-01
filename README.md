# GNN Molecular Property Prediction

A Graph Neural Network (GNN) based system for predicting molecular properties, specifically designed to handle imbalanced datasets in toxicity prediction.

## Features

- **Imbalanced Dataset Support**: Handles severely imbalanced datasets using class weights and oversampling techniques
- **Graph Neural Networks**: Uses PyTorch Geometric for molecular graph representation and learning
- **FastAPI Backend**: Robust REST API with comprehensive error handling
- **Streamlit Frontend**: Interactive web interface for molecular property prediction
- **Batch Processing**: Support for single molecule and batch predictions
- **Docker Deployment**: Containerized deployment with Docker and docker-compose
- **Comprehensive Evaluation**: Detailed metrics for imbalanced dataset evaluation

## Dataset

The system uses the Tox21 dataset, which is highly imbalanced:
- **Negative samples (non-toxic)**: 93.8% (6,351 molecules)
- **Positive samples (toxic)**: 6.2% (423 molecules)

## Installation

### Prerequisites

- Python 3.9+
- Docker and Docker Compose (for containerized deployment)

### Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Test
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Training the Model

### Handle Imbalanced Dataset

Before training, you can balance the dataset or calculate class weights:

1. **Generate class weights** (recommended):
   ```bash
   python data/balance_dataset.py --input data/tox21_sr-p53.csv --method weights
   ```

2. **Create balanced dataset** (alternative):
   ```bash
   python data/balance_dataset.py --input data/tox21_sr-p53.csv --method oversample
   ```

3. **Both approaches**:
   ```bash
   python data/balance_dataset.py --input data/tox21_sr-p53.csv --method both
   ```

### Train the Model

1. **Train with class weights** (recommended for imbalanced data):
   ```bash
   cd backend
   python train_gnn.py --epochs 50
   ```

2. **Train with balanced dataset**:
   ```bash
   cd backend
   python train_gnn.py --epochs 50 --use-balanced
   ```

3. **Train without imbalance handling** (baseline):
   ```bash
   cd backend
   python train_gnn.py --epochs 50 --no-class-weights
   ```

### Evaluate the Model

Run comprehensive evaluation with metrics suitable for imbalanced datasets:

```bash
python backend/eval_gnn.py
```

This provides:
- Overall accuracy and balanced accuracy
- Class-wise precision, recall, and F1-score
- Macro and weighted averages
- AUC-ROC and AUC-PR scores
- Sensitivity and specificity
- Confusion matrix

## Running the Application

### Local Development

1. **Start the backend API**:
   ```bash
   python backend/gnn_api_advanced.py
   ```
   API will be available at: http://localhost:8000

2. **Start the frontend** (in another terminal):
   ```bash
   streamlit run frontend/app_advanced.py
   ```
   Frontend will be available at: http://localhost:8501

### Docker Deployment

1. **Quick start with docker-compose**:
   ```bash
   docker-compose up --build
   ```

2. **Run in background**:
   ```bash
   docker-compose up -d --build
   ```

3. **Stop services**:
   ```bash
   docker-compose down
   ```

### Services

After deployment, the following services will be available:

- **Backend API**: http://localhost:8000
  - Health check: http://localhost:8000/health
  - API documentation: http://localhost:8000/docs
- **Frontend**: http://localhost:8501

## API Usage

### Single Molecule Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"smiles": "CCO"}'
```

### Batch Prediction

```bash
curl -X POST "http://localhost:8000/batch_predict" \
     -H "Content-Type: application/json" \
     -d '{"smiles_list": ["CCO", "CC(=O)O", "c1ccccc1"]}'
```

### CSV Upload

```bash
curl -X POST "http://localhost:8000/csv_predict" \
     -F "file=@molecules.csv"
```

## Understanding Imbalanced Dataset Handling

### Why It Matters

Standard accuracy can be misleading with imbalanced datasets. A model that always predicts "non-toxic" would achieve 93.8% accuracy but would miss all toxic molecules.

### Implemented Solutions

1. **Class Weights**: 
   - Automatically calculated to give higher weight to minority class
   - Class 0 (non-toxic): weight = 0.533
   - Class 1 (toxic): weight = 8.007

2. **Balanced Metrics**:
   - Balanced accuracy: (Sensitivity + Specificity) / 2
   - AUC-PR: More appropriate than AUC-ROC for imbalanced data
   - Per-class precision, recall, and F1-score

3. **Oversampling**:
   - Simple oversampling by duplicating minority class samples
   - Alternative to SMOTE when molecular structure preservation is important

### Model Performance Interpretation

Focus on these metrics for imbalanced datasets:
- **Balanced Accuracy**: Accounts for class imbalance
- **Recall for Class 1 (Sensitivity)**: Ability to detect toxic molecules
- **Precision for Class 1**: Accuracy of toxic predictions
- **AUC-PR**: Area under precision-recall curve
- **F1-Score for Class 1**: Harmonic mean of precision and recall for toxic class

## File Structure

```
Test/
├── backend/                    # FastAPI backend
│   ├── gnn_api.py             # Basic API
│   ├── gnn_api_advanced.py    # Advanced API with batch processing
│   ├── train_gnn.py           # Training script with imbalance handling
│   ├── eval_gnn.py            # Comprehensive evaluation
│   └── gnn_trained.pth        # Trained model (generated)
├── frontend/                   # Streamlit frontend
│   ├── app.py                 # Basic frontend
│   ├── app_advanced.py        # Advanced frontend
│   └── pages/                 # Multi-page app
├── data/                       # Data and utilities
│   ├── balance_dataset.py     # Dataset balancing utility
│   ├── tox21_sr-p53.csv      # Main dataset
│   └── *.csv                  # Generated balanced datasets
├── Dockerfile.backend         # Backend container
├── Dockerfile.frontend        # Frontend container
├── docker-compose.yml         # Multi-service deployment
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Key Features for Imbalanced Data

### Training Features
- ✅ Class weight calculation and application
- ✅ Balanced dataset generation via oversampling
- ✅ Comprehensive evaluation metrics
- ✅ Command-line options for different training modes

### API Features
- ✅ Robust error handling with detailed error messages
- ✅ Input validation and sanitization
- ✅ Batch processing with success/failure tracking
- ✅ CSV upload and processing
- ✅ Health checks and model status monitoring

### Evaluation Features
- ✅ Balanced accuracy calculation
- ✅ Per-class precision, recall, F1-score
- ✅ AUC-ROC and AUC-PR scores
- ✅ Sensitivity and specificity
- ✅ Detailed confusion matrix analysis

## Development

### Adding New Features

1. **Model improvements**: Modify `train_gnn.py` and `eval_gnn.py`
2. **API endpoints**: Add to `gnn_api_advanced.py`
3. **Frontend features**: Modify `app_advanced.py`
4. **Data processing**: Add utilities to `data/` directory

### Running Tests

```bash
# Test dataset balancing
python data/balance_dataset.py --input data/tox21_sr-p53.csv --method both

# Test training
python backend/train_gnn.py --epochs 5

# Test evaluation
python backend/eval_gnn.py

# Test API
python backend/gnn_api_advanced.py &
curl http://localhost:8000/health
```

## Troubleshooting

### Common Issues

1. **Model not found**: Run training first with `python backend/train_gnn.py`
2. **Import errors**: Ensure all dependencies are installed with `pip install -r requirements.txt`
3. **Docker build fails**: Check Docker is running and you have sufficient disk space
4. **API errors**: Check logs with `docker-compose logs backend`

### Performance Tips

1. **For better performance on imbalanced data**:
   - Use class weights rather than oversampling for large datasets
   - Focus on F1-score and AUC-PR rather than accuracy
   - Consider different probability thresholds based on use case

2. **For faster training**:
   - Reduce batch size if memory constrained
   - Use fewer epochs for initial testing
   - Consider using GPU if available

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with appropriate tests
4. Update documentation
5. Submit a pull request

## License

[Add your license information here]