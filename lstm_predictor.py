"""
Backward-compatible wrapper for LSTM predictor.
Redirects to model.lstm_predictor.
"""
from model.lstm_predictor import (
    prepare_sequences,
    build_lstm_model,
    train_lstm,
    predict_future_prices,
    SEQUENCE_LENGTH,
)

__all__ = [
    "prepare_sequences",
    "build_lstm_model",
    "train_lstm",
    "predict_future_prices",
    "SEQUENCE_LENGTH",
]
