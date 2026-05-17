"""FastAPI REST API for ML Model Inference and Training"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import DataPreprocessor
from src.sklearn_models import SklearnModels
from src.tensorflow_models import TensorFlowModels
from src.pytorch_models import PyTorchModels
from config import Config
from logger import setup_logger

# Initialize FastAPI
app = FastAPI(
    title="ML-App API",
    description="Complete Machine Learning Application with Multiple Frameworks",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware - Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

logger = setup_logger(__name__)

# Initialize models
sklearn_models = SklearnModels()
tf_models = TensorFlowModels()
pytorch_models = PyTorchModels()
preprocessor = DataPreprocessor()


# Pydantic Models for request/response
class PredictRequest(BaseModel):
    model_type: str = "sklearn"
    features: List[float]


class BatchPredictRequest(BaseModel):
    model_type: str = "sklearn"
    samples: List[List[float]]


class TrainRequest(BaseModel):
    model_type: str = "sklearn"
    algorithm: str = "random_forest"


class PreprocessRequest(BaseModel):
    features: List[float]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    service: str


class PredictResponse(BaseModel):
    status: str
    model_type: str
    prediction: float
    timestamp: str


class BatchPredictResponse(BaseModel):
    status: str
    model_type: str
    predictions: List[float]
    sample_count: int
    timestamp: str


class TrainResponse(BaseModel):
    status: str
    model_type: str
    algorithm: str
    result: dict
    timestamp: str


class EvaluateResponse(BaseModel):
    status: str
    model_type: str
    metrics: dict
    timestamp: str


class ModelsResponse(BaseModel):
    status: str
    frameworks: dict
    timestamp: str


class StatsResponse(BaseModel):
    status: str
    api_version: str
    service: str
    features: List[str]
    supported_frameworks: List[str]
    cors: str
    public_access: str
    timestamp: str


# Routes
@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "ML-App FastAPI"
    }


@app.post("/api/v1/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a model with specified framework and parameters"""
    try:
        model_type = request.model_type
        algorithm = request.algorithm
        
        if model_type == "sklearn":
            result = sklearn_models.train(algorithm)
        elif model_type == "tensorflow":
            result = tf_models.train()
        elif model_type == "pytorch":
            result = pytorch_models.train()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        logger.info(f"Model trained: {model_type}/{algorithm}")
        return {
            "status": "success",
            "model_type": model_type,
            "algorithm": algorithm,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """Make predictions using trained models"""
    try:
        model_type = request.model_type
        features = request.features
        
        if not features:
            raise HTTPException(status_code=400, detail="Features required")
        
        if model_type == "sklearn":
            prediction = sklearn_models.predict(features)
        elif model_type == "tensorflow":
            prediction = tf_models.predict(features)
        elif model_type == "pytorch":
            prediction = pytorch_models.predict(features)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        logger.info(f"Prediction made using {model_type}")
        return {
            "status": "success",
            "model_type": model_type,
            "prediction": float(prediction[0]) if hasattr(prediction, '__len__') else float(prediction),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/batch-predict", response_model=BatchPredictResponse)
async def batch_predict(request: BatchPredictRequest):
    """Batch predictions for multiple samples"""
    try:
        model_type = request.model_type
        samples = request.samples
        
        if not samples:
            raise HTTPException(status_code=400, detail="Samples required")
        
        if model_type == "sklearn":
            predictions = [sklearn_models.predict(s) for s in samples]
        elif model_type == "tensorflow":
            predictions = [tf_models.predict(s) for s in samples]
        elif model_type == "pytorch":
            predictions = [pytorch_models.predict(s) for s in samples]
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        logger.info(f"Batch prediction completed for {len(samples)} samples")
        return {
            "status": "success",
            "model_type": model_type,
            "predictions": [float(p[0]) if hasattr(p, '__len__') else float(p) for p in predictions],
            "sample_count": len(samples),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/evaluate", response_model=EvaluateResponse)
async def evaluate(request: TrainRequest):
    """Evaluate model performance"""
    try:
        model_type = request.model_type
        
        if model_type == "sklearn":
            metrics = sklearn_models.evaluate()
        elif model_type == "tensorflow":
            metrics = tf_models.evaluate()
        elif model_type == "pytorch":
            metrics = pytorch_models.evaluate()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")
        
        logger.info(f"Model evaluation completed for {model_type}")
        return {
            "status": "success",
            "model_type": model_type,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/models", response_model=ModelsResponse)
async def list_models():
    """List available models and frameworks"""
    return {
        "status": "success",
        "frameworks": {
            "sklearn": ["random_forest", "gradient_boosting", "svm", "logistic_regression"],
            "tensorflow": ["sequential_nn", "functional_nn"],
            "pytorch": ["mlp_classifier", "cnn_classifier"]
        },
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/v1/preprocess")
async def preprocess(request: PreprocessRequest):
    """Preprocess data"""
    try:
        features = request.features
        
        if not features:
            raise HTTPException(status_code=400, detail="Features required")
        
        processed = preprocessor.preprocess([features])
        
        logger.info("Data preprocessing completed")
        return {
            "status": "success",
            "original": features,
            "processed": processed[0].tolist() if hasattr(processed[0], 'tolist') else processed[0],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Preprocessing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats", response_model=StatsResponse)
async def stats():
    """Get API statistics"""
    return {
        "status": "success",
        "api_version": "v1",
        "service": "ML-App FastAPI",
        "features": [
            "Model Training",
            "Single Prediction",
            "Batch Prediction",
            "Model Evaluation",
            "Data Preprocessing",
            "Model Listing"
        ],
        "supported_frameworks": ["sklearn", "tensorflow", "pytorch"],
        "cors": "enabled",
        "public_access": "enabled",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI on port 8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
