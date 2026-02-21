# GreenPull

Analyze ML repositories for carbon emissions and suggest energy-saving optimizations.

Give it a GitHub repo URL with ML training code. It will:
1. Clone the repo and detect the training entrypoint (regex + GPT)
2. Run the training with CodeCarbon to measure baseline emissions
3. Apply an optimization patch (AMP mixed precision or LoRA)
4. Re-run and measure the optimized emissions
5. Report before/after savings (CO2, kWh, water)

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Redis server (`sudo apt install redis-server` or `brew install redis`)

### 2. Install

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure

Edit `backend/.env` and set your OpenAI API key:

```
OPENAI_API_KEY=sk-your-key-here
```

### 4. Run

Start everything (Redis + worker + API server):

```bash
cd backend
bash run.sh
```

Or start each piece manually:

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: RQ worker (processes jobs)
cd backend
PYTHONPATH=. rq worker --url redis://localhost:6379/0

# Terminal 3: FastAPI server
cd backend
PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Submit a repo for analysis

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/username/ml-repo", "patch_type": "amp"}'
```

Response:
```json
{"job_id": "abc-123", "status": "queued", "message": "Job queued. Poll GET /api/jobs/{job_id} for status."}
```

### 6. Poll for results

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

The response progressively fills in as the pipeline runs:
- `status: "cloning"` -> `"analyzing"` -> `"running_baseline"` -> `"patching"` -> `"running_optimized"` -> `"completed"`
- `detection` — which file was detected, what command, what framework
- `baseline` — emissions, energy, duration, hardware info
- `optimized` — same fields after the patch
- `savings` — how much CO2/energy was saved (absolute + percentage)
- `patch_diff` — the unified diff of what was changed

## Debug Mode

Debug mode is on by default (`DEBUG=true` in `.env`). The RQ worker terminal will print detailed logs for every step:

```
12:34:56 INFO     [Pipeline] STEP 2: Detecting training entrypoint
12:34:56 INFO     [Scan] Scanned 23 .py files, found 3 candidates:
12:34:56 INFO       train.py                                  score= 180  patterns=['training_filename', 'backward', 'optimizer_step', ...]
12:34:56 INFO       utils/trainer.py                          score=  85  patterns=['trainer_call', 'model_train']
12:34:57 INFO     [Detect] Analyzing script 1/3: train.py (score=180)
12:34:58 INFO     [Detect] GPT response for train.py:
                   {"summary": "Main training loop for ResNet...", "is_main_entrypoint": "yes", "confidence": "high", ...}
12:34:58 INFO     [Detect] HIGH CONFIDENCE MATCH: train.py
```

Set `DEBUG=false` in `.env` to reduce log verbosity.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Submit repo URL for analysis |
| GET | `/api/jobs/{id}` | Get job status and results |
| GET | `/api/jobs` | List recent jobs |
| GET | `/api/health` | Health check |

### POST /api/analyze body

```json
{
  "repo_url": "https://github.com/user/repo",
  "patch_type": "amp",
  "country_iso_code": "USA",
  "max_training_seconds": 300
}
```

- `patch_type`: `"amp"` (mixed precision), `"lora"` (LoRA), or `"both"`
- `country_iso_code`: 3-letter ISO code for carbon intensity lookup
- `max_training_seconds`: timeout for each training run (default 300)

## Stack

- **API**: FastAPI + SQLite + SQLAlchemy
- **Queue**: Redis + RQ
- **Detection**: Regex heuristics + OpenAI GPT
- **Measurement**: CodeCarbon (OfflineEmissionsTracker)
- **Patching**: GPT-generated AMP/LoRA code transformations
