
import pytest
import numpy as np
import joblib
from fastapi.testclient import TestClient
from main import app

class TestModel:
    def setup_method(self):
        self.model = joblib.load("models/random_forest_v1.pkl")

    def test_model_loaded(self):
        assert self.model is not None

    def test_prediction_shape(self):
        X = np.random.randn(5, 10)
        predictions = self.model.predict(X)
        assert predictions.shape == (5,)

    def test_prediction_binary(self):
        X = np.random.randn(100, 10)
        predictions = self.model.predict(X)
        assert set(predictions).issubset({0, 1})

    def test_prediction_proba_sum(self):
        X = np.random.randn(10, 10)
        probas = self.model.predict_proba(X)
        assert np.allclose(probas.sum(axis=1), 1.0)

class TestAPI:
    def test_root_endpoint(self):
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200

    def test_health_endpoint(self):
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200

    def test_predict_endpoint(self):
        client = TestClient(app)
        payload = {"features": [0.1] * 10}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in [0, 1]
        assert 0 <= data["probability"] <= 1

class TestPreprocessing:
    def test_normalisation(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        X_norm = (X - X.min()) / (X.max() - X.min())
        assert X_norm.min() >= 0
        assert X_norm.max() <= 1

    def test_standardisation(self):
        X = np.random.randn(1000) * 5 + 10
        X_std = (X - X.mean()) / X.std()
        assert abs(X_std.mean()) < 0.01
        assert abs(X_std.std() - 1.0) < 0.01
