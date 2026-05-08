import unittest
from datetime import datetime

from src.api.schemas import (
    PredictRequest,
    PredictResponse,
    PredictionRecord,
    TrainResponse,
    EvaluateResponse,
)


class TestSchemas(unittest.TestCase):

    def test_predict_request_valid(self):
        features = [1.0] * 30
        request = PredictRequest(features=features, model="LOG_REG")
        self.assertEqual(len(request.features), 30)
        self.assertEqual(request.model, "LOG_REG")

    def test_predict_request_invalid_length(self):
        with self.assertRaises(ValueError):
            PredictRequest(features=[1.0] * 10)

    def test_predict_request_default_model(self):
        request = PredictRequest(features=[1.0] * 30)
        self.assertEqual(request.model, "LOG_REG")

    def test_predict_response_fields(self):
        response = PredictResponse(
            prediction_id="test-123",
            prediction=1,
            probability_malignant=0.8,
            probability_benign=0.2,
            created_at=datetime.now(),
            model_version="v1"
        )
        self.assertEqual(response.prediction_id, "test-123")
        self.assertEqual(response.prediction, 1)
        self.assertAlmostEqual(response.probability_malignant, 0.8)

    def test_prediction_record_fields(self):
        record = PredictionRecord(
            prediction_id="test-456",
            features=[1.0] * 30,
            prediction=0,
            probability_malignant=0.3,
            probability_benign=0.7,
            created_at=datetime.now(),
            model_version="v1"
        )
        self.assertEqual(len(record.features), 30)
        self.assertEqual(record.prediction, 0)

    def test_train_response_fields(self):
        response = TrainResponse(
            model="LOG_REG",
            trained=True,
            train_samples=500,
            message="Training complete"
        )
        self.assertTrue(response.trained)
        self.assertEqual(response.train_samples, 500)

    def test_evaluate_response_fields(self):
        response = EvaluateResponse(
            model="LOG_REG",
            accuracy=0.95,
            precision=0.94,
            recall=0.96,
            f1=0.95,
            eval_samples=100,
            message="Evaluation complete"
        )
        self.assertGreaterEqual(response.accuracy, 0.0)
        self.assertLessEqual(response.accuracy, 1.0)
        self.assertEqual(response.eval_samples, 100)


if __name__ == "__main__":
    unittest.main()
