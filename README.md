# Trial Conversion Model

A production-style machine learning project for predicting trial conversion.

## Project Structure

- `src/trial_conversion_model/data.py` — loads training data from PostgreSQL
- `src/trial_conversion_model/features.py` — creates model features
- `src/trial_conversion_model/train.py` — trains and evaluates the XGBoost model
- `models/` — stores generated model artifacts and evaluation metrics
- `data/` — stores generated raw and processed training data
- `.env.example` — template for required environment variables

## Setup

Install dependencies:

```bash
uv sync
```

Create a `.env` file based on `.env.example` and add the database password:

```text
DB_PASSWORD=your_password_here
```

## Train the Model

Run the training pipeline from the project root:

```bash
uv run trial-conversion-model
```

The pipeline:

1. Loads the latest training data from PostgreSQL
2. Saves a raw copy of the training data
3. Creates the model features
4. Saves the processed training data
5. Trains the XGBoost model
6. Calculates ROC-AUC on the test set
7. Saves the trained model and evaluation metrics

## Generated Artifacts

Running the training pipeline creates:

```text
data/
├── 01_raw/
│   └── trials.csv
└── 02_processed/
    └── trials_processed.csv

models/
├── model.pkl
└── metrics.json
```

These generated artifacts are excluded from Git using `.gitignore`.

## Environment Variables

Database credentials are stored locally in `.env` and are not committed to Git.

Use `.env.example` as the template:

```text
DB_PASSWORD=
```