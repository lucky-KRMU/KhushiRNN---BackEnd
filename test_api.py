"""
Unit and integration tests for KhushiRNN backend API.
"""

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["documentation"] == "/docs"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model"] == "KhushiRNN"


def test_predict_positive():
    response = client.post(
        "/predict",
        json={"review": "This movie was absolutely fantastic, brilliant, and I loved every minute!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Positive"
    assert data["score"] > 0.5
    assert data["confidence"] > 0.5
    assert data["token_count"] > 0


def test_predict_negative():
    response = client.post(
        "/predict",
        json={"review": "This movie was terrible, awful, boring, and complete waste of time."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sentiment"] == "Negative"
    assert data["score"] <= 0.5
    assert data["confidence"] > 0.5
    assert data["token_count"] > 0


def test_predict_empty_review():
    response = client.post("/predict", json={"review": "   "})
    assert response.status_code == 400


def test_batch_predict():
    reviews = [
        "Great acting and wonderful script!",
        "terrible horrible awful waste of time",
    ]
    response = client.post("/batch-predict", json={"reviews": reviews})
    assert response.status_code == 200
    data = response.json()
    assert data["total_count"] == 2
    assert len(data["results"]) == 2
    assert data["results"][0]["sentiment"] == "Positive"
    assert data["results"][1]["sentiment"] == "Negative"


if __name__ == "__main__":
    test_root()
    test_health()
    test_predict_positive()
    test_predict_negative()
    test_predict_empty_review()
    test_batch_predict()
    print("All tests passed successfully!")
