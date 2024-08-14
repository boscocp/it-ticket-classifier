import joblib
import numpy as np
from pydantic import BaseModel

from helpers import preprocess_text


class HeathCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


class Ticket(BaseModel):
    """Ticket with description of a specific problem."""

    description: str


class TicketClassification(BaseModel):
    """Ticket with a prediction and its probability."""

    prediction: str
    probability: str


class TicketModel:
    def __init__(self):
        self.model_fname_ = "./machine_learning_training/model.pkl"
        self.model = joblib.load(self.model_fname_)

    def predict(self, text: str):
        data = preprocess_text(text)
        X = np.array([data])
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X).max()
        return prediction, probability
