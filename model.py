"""
Sentiment Analysis RNN Model Inference Module.

Encapsulates weight loading, tokenization, sequence processing,
and forward pass for the KhushiRNN model.
"""

from pathlib import Path
from typing import Dict, List, Tuple
import json
import re
import numpy as np


class SentimentRNN:
    """
    Recurrent Neural Network for Sentiment Classification (Positive / Negative).
    Loads pre-trained weights and vocabulary to run low-latency inference.
    """

    def __init__(self, weights_dir: Path | str | None = None) -> None:
        if weights_dir is None:
            weights_dir = Path(__file__).resolve().parent / "weights"
        else:
            weights_dir = Path(weights_dir)

        if not weights_dir.exists():
            raise FileNotFoundError(f"Weights directory not found at: {weights_dir}")

        # Load weights and biases
        self.bias_hidden = np.load(weights_dir / "BIAS_HIDDEN.npy")
        self.weight_hidden = np.load(weights_dir / "WEIGHT_HIDDEN.npy")
        self.weight_input = np.load(weights_dir / "WEIGHT_INPUT.npy")
        self.weight_output = np.load(weights_dir / "WEIGHT_OUTPUT.npy")
        self.bias_output = np.load(weights_dir / "BIAS_OUTPUT.npy")
        self.vocabulary = np.load(weights_dir / "VOCABULARY.npy")

        # Load token mapping
        with open(weights_dir / "token_id.json", "r", encoding="utf-8") as f:
            self.token_id: Dict[str, int] = json.load(f)

        self.unk_token_id = self.token_id.get("<UNK>", 1)
        self.hidden_dim = self.weight_hidden.shape[0]  # 25
        self.max_tokens = 120

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Lowercases text, trims leading/trailing whitespace, and collapses multi-spaces.
        """
        text = text.lower().strip()
        return re.sub(r"\s+", " ", text)

    @staticmethod
    def tokenize_text(text: str) -> List[str]:
        """
        Tokenizes normalized text using whitespace splitting.
        """
        return text.split()

    @staticmethod
    def sigmoid(x: np.ndarray | float) -> np.ndarray:
        """
        Numerically stable sigmoid function.
        """
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, tokens: List[str]) -> float:
        """
        Iterates over input tokens updating the RNN hidden state,
        then computes the final output score.
        """
        state = np.zeros((self.hidden_dim, 1), dtype=np.float64)

        for word in tokens:
            token = self.token_id.get(word, self.unk_token_id)
            if token >= len(self.vocabulary):
                token = self.unk_token_id

            embedding = self.vocabulary[token].reshape(self.hidden_dim, 1)

            # Recurrent hidden state update: h_t = sigmoid(W_i * x_t + W_h * h_{t-1} + b_h)
            state = self.sigmoid(
                self.weight_input @ embedding + self.weight_hidden @ state + self.bias_hidden
            )

        # Output projection: y = sigmoid(W_o * h_T + b_o)
        output = self.weight_output @ state + self.bias_output
        z = self.sigmoid(output)

        return float(z[0, 0])

    def predict(self, review: str, threshold: float = 0.5) -> Dict[str, object]:
        """
        Runs full sentiment prediction pipeline on raw text review.
        """
        clean_text = self.normalize_text(review)
        tokens = self.tokenize_text(clean_text)
        truncated_tokens = tokens[: self.max_tokens]

        score = self.forward(truncated_tokens)
        sentiment = "Positive" if score > threshold else "Negative"

        # Confidence: distance from decision boundary normalized to [0.5, 1.0]
        confidence = score if sentiment == "Positive" else (1.0 - score)

        return {
            "sentiment": sentiment,
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "token_count": len(truncated_tokens),
            "total_tokens": len(tokens),
            "truncated": len(tokens) > self.max_tokens,
        }

