# KhushiRNN Backend API

A high-performance, lightweight FastAPI backend for sentiment analysis (Positive / Negative review classification) powered by a custom-trained Recurrent Neural Network (RNN) implemented in NumPy.

---

## 🧠 How the Model Works

The model architecture is a vanilla recurrent neural network (Elman RNN) optimized for sequential text processing:

1. **Text Normalization & Tokenization**:
   - Lowercases text, trims edge spaces, collapses consecutive whitespace.
   - Splits on whitespace into tokens (preserves trained punctuation suffixes like `"good."`, `"awesome!"`).
   - Caps sequence length to a maximum of 120 tokens.

2. **Embedding Lookup**:
   - Maps each word token to its vocabulary ID in `token_id.json` (40,000 token vocabulary).
   - Retrieves a 25-dimensional dense word embedding from `VOCABULARY.npy`. Unknown tokens fall back to `<UNK>`.

3. **Recurrent Hidden State Dynamics**:
   - Initial hidden state: $h_0 = \mathbf{0} \in \mathbb{R}^{25 \times 1}$
   - At each time step $t$:
     $$h_t = \sigma(W_{\text{input}} x_t + W_{\text{hidden}} h_{t-1} + b_{\text{hidden}})$$
   - Where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid activation function.

4. **Output Linear Projection & Classification**:
   - Computes output score from the final hidden state $h_T$:
     $$\hat{y} = \sigma(W_{\text{output}} h_T + b_{\text{output}})_0$$
   - Decision boundary:
     - $\hat{y} > 0.5 \implies$ **Positive**
     - $\hat{y} \le 0.5 \implies$ **Negative**

---

## 📁 Repository Structure

```
Backend/
├── weights/
│   ├── BIAS_HIDDEN.npy      # (25, 1) hidden state bias
│   ├── BIAS_OUTPUT.npy      # (25, 1) output bias
│   ├── WEIGHT_HIDDEN.npy    # (25, 25) recurrent hidden-to-hidden weights
│   ├── WEIGHT_INPUT.npy     # (25, 25) input-to-hidden weights
│   ├── WEIGHT_OUTPUT.npy    # (25, 25) hidden-to-output weights
│   ├── VOCABULARY.npy       # (40000, 25) embedding matrix
│   └── token_id.json        # 40,000 token-to-ID mapping
├── model.py                 # Encapsulated, stateless SentimentRNN inference engine
├── main.py                  # FastAPI server with CORS, health check, and endpoints
├── test_api.py              # Automated unit tests for API endpoints
├── requirements.txt         # Pinned production dependencies
├── Procfile                 # Process file for Render / PaaS deployment
├── render.yaml              # Render Blueprint configuration
├── inference-run.py         # Original reference script
├── .gitignore               # Excludes venv, pycache, .DS_Store, logs
└── venv/                    # Local virtual environment
```

---

## 🚀 Local Development Setup

### 1. Activate the Virtual Environment

```bash
cd Backend
source venv/bin/activate
```

*(If creating a new virtual environment from scratch)*:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Development Server

```bash
uvicorn main:app --reload --port 8000
```

The server will be available at:
- **API URL**: [http://localhost:8000](http://localhost:8000)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

### 3. Run Automated Tests

```bash
python test_api.py
```

---

## 📡 API Endpoints

### 1. Predict Single Review

- **Endpoint**: `POST /predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "review": "The food and service were absolutely fantastic!",
    "threshold": 0.5
  }
  ```
- **Response**:
  ```json
  {
    "review": "The food and service were absolutely fantastic!",
    "sentiment": "Positive",
    "score": 0.8142,
    "confidence": 0.8142,
    "token_count": 7,
    "total_tokens": 7,
    "truncated": false
  }
  ```

### 2. Batch Predict Reviews

- **Endpoint**: `POST /batch-predict`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "reviews": [
      "Loved the experience, will come back again!",
      "terrible horrible awful waste of time"
    ],
    "threshold": 0.5
  }
  ```
- **Response**:
  ```json
  {
    "total_count": 2,
    "results": [
      {
        "review": "Loved the experience, will come back again!",
        "sentiment": "Positive",
        "score": 0.7631,
        "confidence": 0.7631,
        "token_count": 7,
        "total_tokens": 7,
        "truncated": false
      },
      {
        "review": "terrible horrible awful waste of time",
        "sentiment": "Negative",
        "score": 0.3048,
        "confidence": 0.6952,
        "token_count": 6,
        "total_tokens": 6,
        "truncated": false
      }
    ]
  }
  ```

### 3. Health Check (Render Uptime Monitoring)

- **Endpoint**: `GET /health`
- **Response**:
  ```json
  {
    "status": "healthy",
    "model": "KhushiRNN",
    "version": "1.0.0",
    "uptime_seconds": 124.5
  }
  ```

---

## 🌐 Frontend Integration (JavaScript / React Example)

```javascript
async function analyzeSentiment(reviewText) {
  const BACKEND_URL = "https://khushirnn-backend.onrender.com"; // or http://localhost:8000

  const response = await fetch(`${BACKEND_URL}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ review: reviewText }),
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.statusText}`);
  }

  const data = await response.json();
  console.log(`Sentiment: ${data.sentiment} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`);
  return data;
}
```

---

## 👤 Author

- **Lucky Pawar** ([@lucky-KRMU](https://github.com/lucky-KRMU))

