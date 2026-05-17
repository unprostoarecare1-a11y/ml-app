# 🤖 ML-App: Complete Machine Learning Application

A production-ready, full-stack machine learning application with multiple frameworks, REST APIs, Docker deployment, and comprehensive documentation. **All endpoints are publicly accessible.**

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [API Documentation](#api-documentation)
- [Installation](#installation)
- [Usage](#usage)
- [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

## ✨ Features

### 🔧 Multiple ML Frameworks
- **scikit-learn**: RandomForest, GradientBoosting, SVM, LogisticRegression
- **TensorFlow/Keras**: Deep learning neural networks
- **PyTorch**: Custom neural network architectures

### 🚀 REST APIs
- **Flask**: Classic REST API on port 5000
- **FastAPI**: Modern async API on port 8000 with auto-documentation
- **CORS Enabled**: All origins allowed (`*`)
- **Public Access**: No authentication required

### 📦 Data Pipeline
- Data preprocessing and scaling
- Feature engineering
- Outlier detection and handling
- Missing value imputation
- Categorical encoding

### 🐳 Docker Support
- Single container deployments
- Docker Compose with 6 services:
  - Flask API
  - FastAPI
  - PostgreSQL database
  - Redis cache
  - PgAdmin (DB management)
  - Training service

### ✅ Complete Testing Suite
- Unit tests for preprocessing
- Model testing
- API endpoint testing
- pytest integration

### 📚 Full Documentation
- Comprehensive README
- API examples
- Configuration guide
- Deployment instructions

## 🏗️ Architecture

```
ml-app/
├── src/
│   ├── preprocessing.py       # Data preprocessing
│   ├── sklearn_models.py      # Scikit-learn models
│   ├── tensorflow_models.py   # TensorFlow models
│   └── pytorch_models.py      # PyTorch models
├── api/
│   ├── flask_app.py          # Flask REST API
│   └── fastapi_app.py        # FastAPI REST API
├── models/
│   └── saved/                 # Trained models
├── data/
│   ├── raw/                   # Raw data
│   └── processed/             # Processed data
├── docker/
│   ├── Dockerfile.flask       # Flask container
│   ├── Dockerfile.fastapi     # FastAPI container
│   └── Dockerfile.training    # Training container
├── tests/                     # Unit tests
├── config.py                  # Configuration
├── logger.py                  # Logging setup
├── main.py                    # CLI interface
├── requirements.txt           # Dependencies
└── docker-compose.yml         # Multi-container setup
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/unprostoarecare1-a11y/ml-app.git
cd ml-app

# Install dependencies
pip install -r requirements.txt

# Copy environment file
cp .env.example .env
```

### 2. Train Models

```bash
# Train all models
python main.py train --model-type all

# Train specific framework
python main.py train --model-type sklearn
python main.py train --model-type tensorflow
python main.py train --model-type pytorch
```

### 3. Start APIs

```bash
# Option 1: Flask (Port 5000)
python main.py api --api-type flask

# Option 2: FastAPI (Port 8000)
python main.py api --api-type fastapi

# Option 3: Both
python main.py api --api-type all
```

### 4. Make Predictions

```bash
# Via command line
python main.py inference --input data.csv --output predictions.csv

# Via API (see API Documentation below)
```

## 🔌 API Documentation

### Flask API (Port 5000)

#### Health Check
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "service": "ML-App Flask API"
}
```

#### Train Model
```bash
curl -X POST http://localhost:5000/api/v1/train \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sklearn",
    "algorithm": "random_forest"
  }'
```

#### Single Prediction
```bash
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sklearn",
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

Response:
```json
{
  "status": "success",
  "model_type": "sklearn",
  "prediction": 0.95,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Batch Predictions
```bash
curl -X POST http://localhost:5000/api/v1/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sklearn",
    "samples": [
      [5.1, 3.5, 1.4, 0.2],
      [7.0, 3.2, 4.7, 1.4],
      [6.3, 3.3, 6.0, 2.5]
    ]
  }'
```

#### Model Evaluation
```bash
curl -X POST http://localhost:5000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"model_type": "sklearn"}'
```

#### List Available Models
```bash
curl http://localhost:5000/api/v1/models
```

#### Data Preprocessing
```bash
curl -X POST http://localhost:5000/api/v1/preprocess \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

#### API Statistics
```bash
curl http://localhost:5000/api/v1/stats
```

### FastAPI (Port 8000)

Same endpoints as Flask, but with:
- **Auto-generated docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Async processing** for better performance
- **Interactive API explorer**

#### Example FastAPI Request
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "sklearn",
    "features": [5.1, 3.5, 1.4, 0.2]
  }'
```

## 💻 Installation

### Prerequisites
- Python 3.8+
- pip
- Docker (optional)
- Docker Compose (optional)

### Step-by-Step Installation

```bash
# 1. Clone the repository
git clone https://github.com/unprostoarecare1-a11y/ml-app.git
cd ml-app

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup environment
cp .env.example .env

# 5. Create necessary directories
mkdir -p data/raw data/processed models/saved logs

# 6. Verify installation
python -c "import tensorflow, torch, sklearn; print('All frameworks installed!')"
```

## 📖 Usage

### Command Line Interface

```bash
# Train models
python main.py train --model-type all
python main.py train --model-type sklearn --algorithm random_forest

# Run inference
python main.py inference --input data.csv --output predictions.csv

# Start API
python main.py api --api-type flask
python main.py api --api-type fastapi

# Docker operations
python main.py docker build
python main.py docker up
python main.py docker down
```

### Python API

```python
from src.sklearn_models import SklearnModels
from src.preprocessing import DataPreprocessor

# Initialize
models = SklearnModels()
preprocessor = DataPreprocessor()

# Preprocess data
data = preprocessor.preprocess(raw_data)

# Train
models.train('random_forest')

# Predict
predictions = models.predict([5.1, 3.5, 1.4, 0.2])

# Evaluate
metrics = models.evaluate()
print(metrics)
```

## 🐳 Docker Deployment

### Build and Run Single Container

```bash
# Build Flask container
docker build -t ml-app-flask -f docker/Dockerfile.flask .

# Run Flask container
docker run -p 5000:5000 -v $(pwd):/app ml-app-flask

# Build FastAPI container
docker build -t ml-app-fastapi -f docker/Dockerfile.fastapi .

# Run FastAPI container
docker run -p 8000:8000 -v $(pwd):/app ml-app-fastapi
```

### Multi-Container Deployment with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Access services
# Flask API: http://localhost:5000
# FastAPI: http://localhost:8000 (Docs: http://localhost:8000/docs)
# PostgreSQL: localhost:5432
# PgAdmin: http://localhost:5050
# Redis: localhost:6379
```

### Service Details

| Service | Port | URL | Credentials |
|---------|------|-----|-------------|
| Flask API | 5000 | http://localhost:5000 | Public |
| FastAPI | 8000 | http://localhost:8000 | Public |
| PostgreSQL | 5432 | localhost:5432 | mluser / mlpassword |
| PgAdmin | 5050 | http://localhost:5050 | admin@mlapp.com / admin |
| Redis | 6379 | localhost:6379 | No auth |

## ⚙️ Configuration

### Environment Variables

Edit `.env` file:

```env
# API Settings
FLASK_ENV=production
FASTAPI_WORKERS=4

# Database
DATABASE_URL=postgresql://mluser:mlpassword@postgres:5432/ml_database

# Cache
REDIS_URL=redis://redis:6379/0

# CORS & Security
ALLOW_ORIGINS=*
CORS_ENABLED=True
REQUIRE_AUTHENTICATION=False

# Model Training
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001

# Logging
LOG_LEVEL=INFO
```

### Configuration File

Edit `config.py`:

```python
class Config:
    DEBUG = False
    TESTING = False
    MODEL_SAVE_PATH = 'models/saved/'
    DATA_PATH = 'data/'
    LOG_LEVEL = 'INFO'
    MAX_BATCH_SIZE = 1000
```

## ✅ Testing

### Run All Tests

```bash
pytest -v

# With coverage
pytest --cov=src tests/

# Specific test file
pytest tests/test_models.py -v

# Specific test function
pytest tests/test_models.py::test_sklearn_training -v
```

### Test Files

- `tests/test_preprocessing.py` - Data preprocessing tests
- `tests/test_models.py` - Model training and evaluation tests
- `tests/test_api.py` - API endpoint tests

## 🔧 Troubleshooting

### Module Import Errors
```bash
# Ensure dependencies are installed
pip install -r requirements.txt --force-reinstall

# Verify Python path
python -c "import sys; print(sys.path)"
```

### Model Not Found
```bash
# Train model first
python main.py train --model-type sklearn

# Verify model file exists
ls -la models/saved/
```

### Port Already in Use
```bash
# Change port in code or use different port
python -m flask run --port 5001
```

### Docker Connection Issues
```bash
# Restart Docker service
docker-compose restart

# Check logs
docker-compose logs postgres redis

# Rebuild containers
docker-compose up --build
```

### Database Connection Error
```bash
# Wait for PostgreSQL to be ready
docker-compose logs postgres

# Check credentials in .env
grep DATABASE_URL .env

# Verify network
docker network ls
```

### GPU/CUDA Issues (TensorFlow/PyTorch)
```bash
# Check CUDA availability
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"

# Fallback to CPU
export CUDA_VISIBLE_DEVICES=""
python main.py train --model-type tensorflow
```

## 📊 Performance Metrics

- **Flask API**: ~100ms average response time
- **FastAPI**: ~50ms average response time
- **Batch Prediction**: 1000 samples in ~2 seconds
- **Model Training**: Depends on dataset size

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🆘 Support

For issues and questions:
- Create an issue on GitHub
- Check existing documentation
- Review error logs in `logs/` directory

## 🎯 Roadmap

- [ ] WebSocket support for real-time predictions
- [ ] Model versioning and rollback
- [ ] Advanced monitoring dashboard
- [ ] Kubernetes deployment templates
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Model explainability (SHAP, LIME)
- [ ] A/B testing framework
- [ ] GraphQL API option

## 📞 Contact

- GitHub: [@unprostoarecare1-a11y](https://github.com/unprostoarecare1-a11y)
- Email: unprostoarecare@gmail.com

---

**Made with ❤️ by ML-App Team**

Last Updated: 2024-01-15
Version: 1.0.0
