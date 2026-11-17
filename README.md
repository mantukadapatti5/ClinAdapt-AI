# ClinAdapt AI (research prototype, not for clinical use)
## Run (Windows PowerShell)
Backend:
    cd backend
    py -3.11 -m venv .venv ; .\.venv\Scripts\Activate.ps1
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install -r requirements.txt
    uvicorn app.main:app --port 8000
Frontend (new terminal): `cd frontend ; npm install ; npm run dev` then open http://localhost:5173
Local LLM: install Ollama and pull `qwen2.5:0.5b` (`qwen2.5:1.5b` needs more RAM). If the GPU has no free VRAM, serve CPU-only: `$env:CUDA_VISIBLE_DEVICES="-1"; ollama serve`. Set `OLLAMA_MODEL` if you pull a different model.
External API: set LLM_API_KEY (and optionally LLM_API_BASE, LLM_API_MODEL). Only structured scores and reference text are sent, never images.
## Model weights
Put trained weights at ml/saved_models/densenet121_clinadapt.pt (3 outputs: Pneumonia, Pneumothorax, Effusion) and optionally calibration.json {"temperature": x}. Without them the app runs with an UNTRAINED demo model and says so.

Train the DenseNet on the NIH ChestX-ray14 image-level labels. For practical training time on Windows, use an isolated Python 3.11 environment with CUDA-enabled PyTorch:

    pip install "numpy<2"
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

    cd backend
    python train_model.py --data-dir "D:\dataset" --epochs 3 --batch-size 16 --workers 0

The script uses the official train/validation list, reserves 10% of its patients for validation, and keeps the official test list out of training. Missing image files are reported and skipped. It downloads the standard ImageNet DenseNet121 initialization if it is not cached, and saves the best validation-loss checkpoint to `ml/saved_models/densenet121_clinadapt.pt`. These weak-label research weights are not clinically validated.

## Datasets
The backend discovers local data from `CLINADAPT_DATA_DIR`, then `data/`, then the sibling `archive (1)` folder. The expected layout:

    data/
      sample_labels.csv            # NIH ChestX-ray14 sample labels
      sample/images/*.png          # 5,606 labelled images
      train_cheXbert.csv           # CheXpert train labels (labels only)
      train_visualCheXbert.csv     # CheXpert train labels (labels only)

CheXpert label tables are browsable and verifiable, but their JPGs are not part of this download, so only the NIH images can be analysed from the dataset browser. Dataset labels are text-mined research annotations, not confirmed diagnoses.

Endpoints: `GET /api/datasets`, `/api/datasets/{id}/stats`, `/items`, `/preview`, `/thumbnail/{image}`, `/image/{image}`, `GET /api/datasets/verify`; `POST /api/analyze` accepts `dataset` + `image_id` instead of a file and returns the dataset ground truth with the prediction.

Verify the local bundles (checks row counts, image/label joins, readable images, label domains, and CSV alignment):

    cd backend
    python verify_datasets.py

## Docker
Docker Desktop (Linux containers) serves the API and the built UI from one container:

    cd frontend && npm ci && npm run build && cd ..
    docker compose up --build -d
    # UI and API http://localhost:8000
    docker compose logs -f backend

Compose bind-mounts the dataset archive read-only at /datasets (override with `CLINADAPT_DATA_DIR`), `ml/saved_models` for weights, the backend code, and `frontend/dist`. SQLite case history lives in the `clinadapt_state` volume; rerun `npm run build` after frontend changes. Ollama on the host is reachable at `host.docker.internal:11434`; compose defaults `OLLAMA_MODEL` to `qwen2.5:0.5b`. Without Ollama, explanations fall back to templates.
