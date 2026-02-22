# GreenPull

Analyze ML repositories for carbon emissions and suggest energy-saving optimizations.

Give it a GitHub repo URL with ML training code. It will:

1. Clone the repo and detect the training entrypoint (regex heuristics + GPT)
2. Extract training configuration (model architecture, params, epochs, hardware) via GPT analysis of source code, config files, and imported modules
3. Estimate baseline carbon emissions using [Green Algorithms](https://www.green-algorithms.org/) formulas
4. Generate an optimization patch (AMP mixed precision and/or LoRA)
5. Estimate optimized emissions with the patch applied
6. Report savings with real-world comparisons (tree-months, car km, smartphone charges, etc.)

No code is executed. All estimation is done via static analysis + Green Algorithms formulas.

## How It Works

```text
Clone repo
    |
    v
Detect training entrypoint (regex scoring + iterative GPT analysis)
    |
    v
Gather context (config files, imported modules, README, dependencies)
    |
    v
GPT extracts training config (model type, parameter count, epochs, GPU needs, runtime estimate)
    |
    v
Estimate baseline emissions (Green Algorithms: power = PUE * (CPU + GPU + memory), energy = power * hours, carbon = energy * country_CI)
    |
    v
GPT generates AMP/LoRA patch (preview only, original code unchanged)
    |
    v
GPT estimates optimized runtime/memory
    |
    v
Estimate optimized emissions -> compute savings + comparisons
```

## Quick Start

### 1. Prerequisites

- Python 3.10+
- Redis server (`sudo apt install redis-server` or `brew install redis`)

### 2. Install

```bash
conda create -n greenpull python=3.11 -y
conda activate greenpull
cd backend
pip install -r requirements.txt
```

### 3. Configure

Create `backend/.env`:

```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4o
DEBUG=true
```

### 4. Run

Start everything (Redis + worker + API server):

```bash
cd backend
bash run.sh
```

Or manually:

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: RQ worker
cd backend && PYTHONPATH=. rq worker --url redis://localhost:6379/0

# Terminal 3: FastAPI
cd backend && PYTHONPATH=. uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Submit a repo

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"repo_url": "https://github.com/user/ml-repo", "patch_type": "amp", "country_iso_code": "DEU"}'
```

### 6. Poll for results

```bash
curl http://localhost:8000/api/jobs/<job_id>
```

Status progression: `queued` -> `cloning` -> `analyzing` -> `extracting_config` -> `estimating_baseline` -> `patching` -> `estimating_optimized` -> `completed`

Response includes:

- `detection` -- entrypoint file, framework, run command
- `training_config` -- model type, parameter count, epochs, batch size, GPU type, estimated runtime
- `baseline` / `optimized` -- emissions (kg CO2), energy (kWh), duration, CPU/GPU energy breakdown
- `savings` -- absolute and percentage reductions
- `comparisons` -- tree-months, car km, smartphone charges, streaming hours, flight fraction, LED bulb hours
- `patch_diff` -- unified diff of the optimization patch

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze` | Submit repo URL for analysis |
| GET | `/api/jobs/{id}` | Get job status and results |
| GET | `/api/jobs` | List recent jobs |
| GET | `/api/health` | Health check |

### POST /api/analyze

```json
{
  "repo_url": "https://github.com/user/repo",
  "patch_type": "amp",
  "country_iso_code": "DEU"
}
```

- `patch_type`: `"amp"` (mixed precision), `"lora"` (LoRA), or `"both"`
- `country_iso_code`: ISO 3166-1 alpha-3 code for carbon intensity (default: `"DEU"`)

Supported countries include all EU members + USA, CAN, BRA, CHN, IND, JPN, KOR, AUS, and more. Falls back to world average if unknown.

## Estimation Method

Based on [Green Algorithms](https://www.green-algorithms.org/) (Lannelongue et al., 2021):

```text
power (W)  = PUE * (CPU_cores * TDP/core * usage + GPUs * GPU_TDP * usage + memory_GB * 0.375)
energy (kWh) = power * runtime_hours / 1000
carbon (g CO2) = energy * carbon_intensity(country)
```

- GPU TDP data for 25+ GPUs (H100, A100, V100, T4, RTX series, etc.)
- CPU TDP for 10 classes (Xeon, EPYC, Ryzen, desktop, cloud)
- Carbon intensity for 30+ countries from CodeCarbon/IEA data
- PUE: 1.0 (local) or 1.2 (cloud)

## Frontend

The web UI is a React + TypeScript app built with Vite, Tailwind CSS, and shadcn/ui.

### Prerequisites

- Node.js 18+ (for `node:` protocol imports)
- npm, yarn, or bun

### Install & Run

```bash
cd frontend
npm install
npm run dev
```

The dev server starts on `http://localhost:8080` and proxies `/api/*` requests to the backend at `localhost:8000`. Make sure the backend is running first.

### Build for Production

```bash
cd frontend
npm run build
npm run preview
```

## Stack

- **Frontend**: React 18 + TypeScript + Vite + Tailwind CSS + shadcn/ui + Recharts
- **API**: FastAPI + SQLite + SQLAlchemy
- **Queue**: Redis + RQ
- **Detection**: Regex heuristics + OpenAI GPT (iterative analysis)
- **Estimation**: Green Algorithms formulas
- **Patching**: GPT-generated AMP/LoRA code transformations (preview only)
