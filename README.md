# Hand Sign Detection Dynamic

A full-stack hand sign recognition system built for practical experimentation and production-style iteration. It combines browser-based inference, profile-aware local training, and a shared artifact contract that keeps training and serving cleanly decoupled.

## Why This Project Is Useful

- Real-time sign prediction from webcam frames through a modern Next.js frontend.
- Dual modeling strategy: Random Forest for low-latency static gestures and LSTM for sequence-aware dynamic signs.
- Built-in combo detection that recognizes gesture phrases from a rolling prediction stream.
- Multiple training paths: browser-triggered API training, local device CLI workflow, and root orchestration pipeline.
- Shared backend state registry so active artifacts are consistently discoverable.

## Core Capabilities

| Capability | What It Does | Primary Files |
|---|---|---|
| Live Inference | Captures camera frames and returns label + confidence in near real time | `frontend/app/page.tsx`, `src/api_server.py` |
| Static Model Path | Trains/serves Random Forest from landmark-style features | `src/random_forest_trainer.py`, `src/api_server.py` |
| Dynamic Model Path | Trains/serves LSTM from fixed-length feature sequences | `src/wlasl_data_preprocessor.py`, `src/lstm_trainer.py` |
| Combo Layer | Matches recent predictions to predefined gesture templates | `src/api_server.py` |
| Device-Local Training | Profile-aware CLI for constrained hardware and packaging | `src/training_pipeline.py` |
| Shared Artifact Contract | Publishes active model/data paths for backend consumption | `src/shared_artifacts.py`, `models/shared_backend_state.json` |

## 90-Second Quickstart

### 1. Install Python dependencies

```bash
pip install -r requirements-runtime.txt
```

For full training workflows:

```bash
pip install -r requirements-training.txt
```

### 2. Start the FastAPI backend

```bash
python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Start the Next.js frontend

```bash
cd frontend
npm install
npm.cmd run dev
```

If PowerShell blocks npm scripts, use `npm.cmd` instead of `npm`.

### 4. Open the app

- Frontend: `http://127.0.0.1:3000`
- Backend docs: `http://127.0.0.1:8000/docs`

## Start Path Guide

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0b1020',
  'primaryColor': '#1f2937',
  'primaryTextColor': '#f9fafb',
  'primaryBorderColor': '#38bdf8',
  'lineColor': '#38bdf8',
  'secondaryColor': '#0f172a',
  'tertiaryColor': '#111827'
}}}%%
flowchart TD
  A[Open Repository] --> B{Your Goal}
  B -->|Run Demo| C[Start Backend and Frontend]
  B -->|Train on Device| D[Use training_pipeline command mode]
  B -->|Train from UI/API| E[Use /train, /train_csv, /train_lstm]
  B -->|Understand Design| F[Read architecture_and_workflows.md]
  C --> G[Live Prediction + Combo Feedback]
  D --> H[Package Artifacts + Update Shared State]
  E --> H
  H --> G

  classDef decision fill:#fef08a,stroke:#eab308,color:#1f2937,stroke-width:2px;
  classDef action fill:#22d3ee,stroke:#0891b2,color:#082f49,stroke-width:2px;
  classDef deep fill:#a78bfa,stroke:#7c3aed,color:#1e1b4b,stroke-width:2px;
  classDef result fill:#34d399,stroke:#059669,color:#052e16,stroke-width:2px;
  class A,C,D,E,F action;
  class B decision;
  class H deep;
  class G result;
```

## System Architecture

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0a0f1f',
  'primaryColor': '#111827',
  'primaryTextColor': '#f8fafc',
  'primaryBorderColor': '#22d3ee',
  'lineColor': '#38bdf8',
  'clusterBkg': '#0f172a',
  'clusterBorder': '#334155'
}}}%%
flowchart LR
  subgraph Data
    CSV[CSV Landmark Data]
    WLASL[WLASL JSON + Videos]
  end

  subgraph Training
    RFTRAIN[RF Training]
    PREP[Sequence Preprocessing]
    LSTMTRAIN[LSTM Training]
    PACKAGE[Artifact Packaging]
  end

  subgraph Runtime
    UI[Next.js Frontend]
    API[FastAPI Backend]
    RF[Random Forest Model]
    LSTM[LSTM Model]
    COMBO[Combo Detector]
  end

  CSV --> RFTRAIN
  WLASL --> PREP --> LSTMTRAIN
  RFTRAIN --> PACKAGE
  LSTMTRAIN --> PACKAGE
  PACKAGE --> STATE[shared_backend_state.json]
  STATE --> API
  API --> RF
  API --> LSTM
  API --> COMBO
  UI --> API
  API --> UI

  classDef data fill:#0ea5e9,stroke:#0369a1,color:#082f49,stroke-width:2px;
  classDef train fill:#c084fc,stroke:#7e22ce,color:#3b0764,stroke-width:2px;
  classDef runtime fill:#34d399,stroke:#059669,color:#052e16,stroke-width:2px;
  classDef state fill:#f97316,stroke:#c2410c,color:#431407,stroke-width:2px;
  class CSV,WLASL data;
  class RFTRAIN,PREP,LSTMTRAIN,PACKAGE train;
  class UI,API,RF,LSTM,COMBO runtime;
  class STATE state;
```

## Inference Runtime Workflow

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0b1020',
  'primaryColor': '#0f172a',
  'primaryTextColor': '#e2e8f0',
  'lineColor': '#22d3ee',
  'actorBkg': '#1d4ed8',
  'actorBorder': '#93c5fd',
  'actorTextColor': '#eff6ff',
  'signalColor': '#22d3ee',
  'signalTextColor': '#e0f2fe',
  'labelBoxBkgColor': '#111827',
  'labelBoxBorderColor': '#334155',
  'labelTextColor': '#f8fafc'
}}}%%
sequenceDiagram
  participant User
  participant Frontend
  participant Backend
  participant Model
  participant Combo

  User->>Frontend: Start camera and perform sign
  Frontend->>Backend: POST image(s) to /predict or /predict_sequence
  Backend->>Model: Run RF or LSTM inference
  Model-->>Backend: label + probability
  Backend->>Combo: Update rolling buffer
  Combo-->>Backend: combo hit or miss
  Backend-->>Frontend: JSON response
  Frontend-->>User: Render label, confidence, combo status
```

## Static vs Dynamic Pipeline

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#090f1a',
  'primaryColor': '#111827',
  'primaryTextColor': '#f8fafc',
  'primaryBorderColor': '#22d3ee',
  'lineColor': '#38bdf8',
  'clusterBkg': '#111827',
  'clusterBorder': '#334155'
}}}%%
flowchart LR
  subgraph Static_Path
    S1[hand_alphabet_data.csv]
    S2[Feature Matrix]
    S3[RandomForest Training]
    S4[hand_alphabet_model.pkl + class_labels.npy]
  end

  subgraph Dynamic_Path
    D1[WLASL_v0.3.json + videos]
    D2[Frame and Feature Extraction]
    D3[Sequence Builder]
    D4[X_data.npy + y_data.npy]
    D5[LSTM Training]
    D6[gesture_model.h5 + wlasl_labels.npy]
  end

  S1 --> S2 --> S3 --> S4
  D1 --> D2 --> D3 --> D4 --> D5 --> D6

  classDef static fill:#06b6d4,stroke:#0e7490,color:#083344,stroke-width:2px;
  classDef dynamic fill:#a78bfa,stroke:#7c3aed,color:#2e1065,stroke-width:2px;
  classDef artifact fill:#f59e0b,stroke:#b45309,color:#451a03,stroke-width:2px;
  class S1,S2,S3 static;
  class D1,D2,D3,D5 dynamic;
  class S4,D4,D6 artifact;
```

## Training Workflow Decision Tree

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0b1120',
  'primaryColor': '#111827',
  'primaryTextColor': '#f9fafb',
  'primaryBorderColor': '#f472b6',
  'lineColor': '#22d3ee'
}}}%%
flowchart TD
  A[Need to Retrain] --> B{Where are you training?}
  B -->|Constrained device| C[Profile: pi_zero]
  B -->|Workstation| D[Profile: full]

  C --> E[preprocess]
  C --> F[train-rf]
  C --> G[evaluate]
  C --> H[package]
  C --> I[device-all]

  D --> J[preprocess with higher limits]
  D --> K[train-rf or legacy all]
  D --> L[train_lstm path]

  E --> M[shared_backend_state.json updated]
  F --> M
  G --> M
  H --> M
  I --> M
  J --> M
  K --> M
  L --> M

  classDef decision fill:#fde68a,stroke:#f59e0b,color:#422006,stroke-width:2px;
  classDef profile fill:#f9a8d4,stroke:#db2777,color:#500724,stroke-width:2px;
  classDef action fill:#67e8f9,stroke:#0891b2,color:#083344,stroke-width:2px;
  classDef result fill:#86efac,stroke:#16a34a,color:#052e16,stroke-width:2px;
  class A,B decision;
  class C,D profile;
  class E,F,G,H,I,J,K,L action;
  class M result;
```

## Artifact Lifecycle

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0a1222',
  'primaryColor': '#111827',
  'primaryTextColor': '#f8fafc',
  'primaryBorderColor': '#22d3ee',
  'lineColor': '#38bdf8'
}}}%%
flowchart LR
  A[Training Command or API Trigger] --> B[Create Model and Label Artifacts]
  B --> C[Publish Active Paths]
  C --> D[models/shared_backend_state.json]
  D --> E[Backend Artifact Resolver]
  E --> F[Runtime Inference]
  F --> G[Frontend Prediction UI]

  classDef step fill:#67e8f9,stroke:#0891b2,color:#083344,stroke-width:2px;
  classDef artifact fill:#fca5a5,stroke:#dc2626,color:#450a0a,stroke-width:2px;
  classDef final fill:#86efac,stroke:#16a34a,color:#052e16,stroke-width:2px;
  class A,B,C,E,F step;
  class D artifact;
  class G final;
```

## Run and Train Commands

### Backend + Frontend

```bash
# Terminal 1
python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2
cd frontend
npm.cmd run dev
```

### Device-local trainer commands

```bash
# End-to-end local flow
python src/training_pipeline.py --command device-all --profile pi_zero --note "local run"

# Individual steps
python src/training_pipeline.py --command preprocess --profile pi_zero
python src/training_pipeline.py --command train-rf --profile pi_zero
python src/training_pipeline.py --command evaluate --profile pi_zero
python src/training_pipeline.py --command package --profile pi_zero
```

### Useful overrides for preprocessing

```bash
python src/training_pipeline.py --command preprocess --profile pi_zero --max-classes 12 --max-videos-per-class 4 --sequence-length 24 --frame-stride 2
```

### Legacy mode

```bash
python src/training_pipeline.py --model all
python src/training_pipeline.py --model random_forest
python src/training_pipeline.py --model lstm --data wlasl
```

### Root orchestrator

```bash
python model_training_orchestrator.py
```

## API Endpoint Summary

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/predict` | Single-frame prediction using Random Forest |
| POST | `/predict_sequence` | Sequence prediction using LSTM (expects 30 frames) |
| GET | `/combos` | List available combo templates and patterns |
| POST | `/clear_combos` | Clear combo buffer state |
| GET | `/` | Backend status with frontend URL hint |
| GET | `/artifacts` | Return active artifact registry |
| GET | `/training` | Training UI guidance endpoint |
| POST | `/train` | Train RF from uploaded image samples + labels |
| POST | `/train_csv` | Train RF from uploaded CSV |
| POST | `/process_wlasl` | Trigger WLASL preprocessing script |
| POST | `/train_lstm` | Trigger LSTM training script |

## Hardware Profiles

| Profile | Intended Hardware | Typical Usage |
|---|---|---|
| `pi_zero` | Raspberry Pi Zero 2 W or similar | On-device preprocessing, RF retraining, packaging |
| `full` | Workstation/laptop with stronger CPU/RAM | Larger preprocessing runs, broader experimentation, LSTM workflows |

## Repository Workflow Map

```mermaid
%%{init: {'theme': 'base', 'themeVariables': {
  'background': '#0b1020',
  'primaryColor': '#111827',
  'primaryTextColor': '#f8fafc',
  'primaryBorderColor': '#38bdf8',
  'lineColor': '#22d3ee',
  'clusterBkg': '#0f172a',
  'clusterBorder': '#334155'
}}}%%
flowchart TD
  A[src/] --> A1[api_server.py]
  A --> A2[training_pipeline.py]
  A --> A3[wlasl_data_preprocessor.py]
  A --> A4[lstm_trainer.py]
  A --> A5[random_forest_trainer.py]

  B[data/] --> B1[CSV and WLASL metadata]
  B --> B2[videos/]
  B --> B3[X_data.npy and y_data.npy]

  C[models/] --> C1[RF and LSTM artifacts]
  C --> C2[class label files]
  C --> C3[shared_backend_state.json]

  D[frontend/] --> D1[app/]
  D --> D2[components/]
  D --> D3[fonts and styling]

  A2 --> C3
  C3 --> A1
  D --> A1

  classDef source fill:#60a5fa,stroke:#1d4ed8,color:#172554,stroke-width:2px;
  classDef data fill:#22d3ee,stroke:#0891b2,color:#083344,stroke-width:2px;
  classDef model fill:#f59e0b,stroke:#b45309,color:#451a03,stroke-width:2px;
  classDef front fill:#a78bfa,stroke:#7c3aed,color:#2e1065,stroke-width:2px;
  class A,A1,A2,A3,A4,A5 source;
  class B,B1,B2,B3 data;
  class C,C1,C2,C3 model;
  class D,D1,D2,D3 front;
```

## Troubleshooting

### Frontend does not start in PowerShell

If script policy blocks npm, run via command shell:

```bash
cmd /c "cd frontend && npm.cmd run dev"
```

### Backend starts but file uploads fail

Install multipart support:

```bash
pip install python-multipart
```

### MediaPipe is unavailable

The backend includes a fallback feature extraction path based on grayscale histogram features, so inference can still run in constrained setups.

### TensorFlow warnings on Windows

GPU-related warnings are common on native Windows setups. CPU training and inference continue to work.

## Documentation and Deep Dives

- System and workflow deep dive: `architecture_and_workflows.md`
- Local trainer operations: `training_guide.md`
- FastAPI implementation: `src/api_server.py`
- Device trainer CLI implementation: `src/training_pipeline.py`

## Contributing

Contributions are welcome in model quality, data pipeline reliability, combo logic, and frontend usability. If you open a PR, include the command path you validated (`run`, `train-rf`, `device-all`, or API training endpoint) and any artifact changes produced.
