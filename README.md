# CoACD Backend Service

Python backend for convex decomposition using the [CoACD library](https://github.com/SarahWeiii/CoACD).

## Local Development

### Prerequisites
- Python 3.11+
- pip

### Setup

```bash
cd coacd-backend
pip install -r requirements.txt
```

### Run locally

```bash
python main.py
```

Or with uvicorn:
```bash
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

- `GET /` - Health check
- `GET /health` - Health check for Render
- `POST /decompose` - Perform convex decomposition

### Test the API

```bash
curl http://localhost:8000/health
```

## Deploy to Render (Free Tier)

### Option 1: Using render.yaml (Blueprint)

1. Push this `coacd-backend` folder to a Git repository
2. Go to [Render Dashboard](https://dashboard.render.com)
3. Click "New" → "Blueprint"
4. Connect your repository
5. Render will detect `render.yaml` and deploy automatically

### Option 2: Manual Deployment

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your repository
4. Configure:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

### Option 3: Docker Deployment

If the native Python deployment has issues with CoACD dependencies:

1. Go to Render Dashboard
2. Click "New" → "Web Service"
3. Select "Docker" as the environment
4. Render will use the included `Dockerfile`

## Frontend Configuration

After deploying, update your frontend environment:

```bash
# In simready-inspector/.env
VITE_COACD_API_URL=https://your-coacd-api.onrender.com
```

Or set it in your deployment platform's environment variables.

## API Usage

### POST /decompose

Request body:
```json
{
  "mesh": {
    "vertices": [x1, y1, z1, x2, y2, z2, ...],
    "indices": [i1, i2, i3, ...]
  },
  "params": {
    "threshold": 0.05,
    "max_convex_hull": -1,
    "preprocess_mode": "auto",
    "preprocess_resolution": 50,
    "mcts_iterations": 100,
    "mcts_depth": 3,
    "mcts_nodes": 20,
    "resolution": 2000,
    "seed": 0
  }
}
```

Response:
```json
{
  "hulls": [
    {
      "id": "hull_...",
      "vertices": [...],
      "indices": [...],
      "volume": 0.123,
      "centroid": [x, y, z],
      "bounding_box": {"min": [...], "max": [...]}
    }
  ],
  "total_volume": 0.5,
  "original_volume": 0.48,
  "volume_error": 4.16,
  "compute_time_ms": 250.5,
  "params": {...}
}
```

## Parameters (from CoACD documentation)

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| threshold | 0.01-1.0 | 0.05 | Concavity threshold for decomposition |
| max_convex_hull | -1 to N | -1 | Max hulls (-1 = no limit) |
| preprocess_mode | auto/on/off | auto | Manifold preprocessing |
| preprocess_resolution | 20-100 | 50 | Preprocess detail level |
| mcts_iterations | 60-2000 | 100 | MCTS search iterations |
| mcts_depth | 2-7 | 3 | MCTS search depth |
| mcts_nodes | 10-40 | 20 | MCTS child nodes |
| resolution | 1000-10000 | 2000 | Sampling resolution |
| seed | 0+ | 0 | Random seed (0 = random) |
# coacd-backend
